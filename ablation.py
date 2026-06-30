import argparse
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import count_params

BLOCK_SIZE = 64
N_EMBD     = 128
DROPOUT    = 0.1
BATCH_SIZE = 16
LR         = 3e-4
EVAL_ITERS = 50

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# ---------------------------------------------------------------------------
# Data (same Shakespeare corpus as train.py)
# ---------------------------------------------------------------------------

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars      = sorted(set(text))
vocab_size = len(chars)
stoi       = {ch: i for i, ch in enumerate(chars)}

def _encode(s):
    return [stoi[c] for c in s]

data       = torch.tensor(_encode(text), dtype=torch.long)
_split     = int(0.9 * len(data))
train_data = data[:_split]
val_data   = data[_split:]


def get_batch(split):
    src = train_data if split == 'train' else val_data
    ix  = torch.randint(len(src) - BLOCK_SIZE, (BATCH_SIZE,))
    x   = torch.stack([src[i:i + BLOCK_SIZE]         for i in ix])
    y   = torch.stack([src[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ---------------------------------------------------------------------------
# Model classes (copied from model.py and parameterised for ablation)
# ---------------------------------------------------------------------------

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, _ = x.shape
        head_size = self.key.out_features
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        scale = head_size ** -0.5
        wei = q @ k.transpose(-2, -1) * scale
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        attn_weights = wei
        wei = self.dropout(wei)
        out = wei @ v
        return out, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(n_heads)])
        self.proj    = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        head_outs, head_attns = [], []
        for head in self.heads:
            out, attn = head(x)
            head_outs.append(out)
            head_attns.append(attn)
        out    = torch.cat(head_outs, dim=-1)
        out    = self.dropout(self.proj(out))
        stacked = torch.stack(head_attns, dim=1)
        return out, stacked


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_heads, n_embd, block_size, has_attention=True):
        super().__init__()
        self.has_attention = has_attention
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        if has_attention:
            head_size = n_embd // n_heads
            self.sa = MultiHeadAttention(n_heads, head_size, n_embd, block_size)
        self.ffn = FeedForward(n_embd)

    def forward(self, x):
        if self.has_attention:
            attn_out, attn_weights = self.sa(self.ln1(x))
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            return x, attn_weights
        else:
            x = x + self.ffn(self.ln2(self.ln1(x)))
            return x, None


class GPT(nn.Module):
    def __init__(self, vocab_size, n_heads=4, n_layers=4,
                 n_embd=N_EMBD, block_size=BLOCK_SIZE,
                 has_attention=True, has_pos_emb=True):
        super().__init__()
        self.has_pos_emb = has_pos_emb
        self.block_size  = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        if has_pos_emb:
            self.positional_embedding = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.ModuleList(
            [Block(n_heads, n_embd, block_size, has_attention) for _ in range(n_layers)]
        )
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        if self.has_pos_emb:
            pos_emb = self.positional_embedding(torch.arange(T, device=idx.device))
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block(x)
            all_attn_weights.append(attn_weights)

        x      = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss, all_attn_weights


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    results = {}
    for split in ('train', 'val'):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss, _ = model(x, y)
            losses[k] = loss.item()
        results[split] = losses.mean().item()
    model.train()
    return results


def train_variant(cfg, n_iters, print_every):
    model = GPT(
        vocab_size     = vocab_size,
        n_heads        = cfg['n_heads'],
        n_layers       = cfg['n_layers'],
        has_attention  = cfg['has_attention'],
        has_pos_emb    = cfg['has_pos_emb'],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"\n{'=' * 60}")
    print(f"Variant : {cfg['name']}")
    print(f"Params  : {count_params(model)}  |  device: {DEVICE}")
    print(f"{'=' * 60}")

    for step in range(n_iters):
        if step % print_every == 0 or step == n_iters - 1:
            losses = estimate_loss(model)
            print(f"  step {step:4d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        x, y = get_batch('train')
        _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final = estimate_loss(model)
    val_loss   = final['val']
    perplexity = math.exp(val_loss)

    return {
        'name':       cfg['name'],
        'val_loss':   round(val_loss, 4),
        'perplexity': round(perplexity, 2),
        'n_params':   count_params(model),
    }


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

CONFIGS = [
    {'name': 'Full model',         'n_heads': 4, 'n_layers': 4, 'has_attention': True,  'has_pos_emb': True},
    {'name': 'Single-head',        'n_heads': 1, 'n_layers': 4, 'has_attention': True,  'has_pos_emb': True},
    {'name': 'No positional emb',  'n_heads': 4, 'n_layers': 4, 'has_attention': True,  'has_pos_emb': False},
    {'name': 'No attention (MLP)', 'n_heads': 4, 'n_layers': 4, 'has_attention': False, 'has_pos_emb': True},
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation study for nano-GPT')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run: 300 iters per variant, print every 50')
    args = parser.parse_args()

    n_iters     = 300 if args.quick else 1500
    print_every = 50  if args.quick else 300

    results = []
    for cfg in CONFIGS:
        result = train_variant(cfg, n_iters, print_every)
        results.append(result)

    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Formatted comparison table
    col = {'name': 25, 'loss': 10, 'ppl': 12, 'params': 10}
    header = (f"{'Variant':<{col['name']}} {'Val Loss':>{col['loss']}} "
              f"{'Perplexity':>{col['ppl']}} {'Params':>{col['params']}}")
    divider = '-' * (col['name'] + col['loss'] + col['ppl'] + col['params'] + 3)
    print(f"\n{header}")
    print(divider)
    for r in results:
        print(f"{r['name']:<{col['name']}} {r['val_loss']:>{col['loss']}.4f} "
              f"{r['perplexity']:>{col['ppl']}.1f} {r['n_params']:>{col['params']}}")

    print(f"\nSaved → ablation_results.json")
