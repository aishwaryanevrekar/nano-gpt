import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK_SIZE = 64
N_EMBD    = 128
N_HEADS   = 4
N_LAYERS  = 4
DROPOUT   = 0.1


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, _ = x.shape
        head_size = self.key.out_features
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        scale = head_size ** -0.5
        wei = q @ k.transpose(-2, -1) * scale              # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        attn_weights = wei                                  # (B, T, T)  pre-dropout
        wei = self.dropout(wei)
        out = wei @ v                                       # (B, T, head_size)
        return out, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj  = nn.Linear(n_heads * head_size, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        head_outs, head_attns = [], []
        for head in self.heads:
            out, attn = head(x)
            head_outs.append(out)
            head_attns.append(attn)

        out = torch.cat(head_outs, dim=-1)          # (B, T, n_heads*head_size)
        out = self.dropout(self.proj(out))           # (B, T, N_EMBD)
        stacked_attn_weights = torch.stack(head_attns, dim=1)  # (B, n_heads, T, T)
        return out, stacked_attn_weights


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = N_EMBD // N_HEADS
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.sa  = MultiHeadAttention(N_HEADS, head_size)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.ffn = FeedForward()

    def forward(self, x):
        attn_out, attn_weights = self.sa(self.ln1(x))
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, N_EMBD)
        self.positional_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f   = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                              # (B, T, N_EMBD)
        pos_emb = self.positional_embedding(torch.arange(T, device=idx.device))  # (T, N_EMBD)
        x = tok_emb + pos_emb

        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block(x)
            all_attn_weights.append(attn_weights)

        x = self.ln_f(x)
        logits = self.lm_head(x)                                         # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss, all_attn_weights

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]             # crop to context window
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature     # (B, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def count_params(model) -> str:
    total = sum(p.numel() for p in model.parameters())
    if total >= 1_000_000:
        return f"{total / 1_000_000:.2f}M"
    if total >= 1_000:
        return f"{total / 1_000:.2f}K"
    return str(total)


if __name__ == '__main__':
    vocab_size = 65
    model = GPT(vocab_size)
    print(f"Parameter count: {count_params(model)}")

    B, T = 2, BLOCK_SIZE
    idx     = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))

    logits, loss, all_attn_weights = model(idx, targets)

    assert logits.shape == (B, T, vocab_size), \
        f"logits shape mismatch: {logits.shape}"
    assert loss is not None, \
        "loss should not be None when targets are provided"
    assert len(all_attn_weights) == N_LAYERS, \
        f"expected {N_LAYERS} attn weight tensors, got {len(all_attn_weights)}"
    assert all_attn_weights[0].shape == (B, N_HEADS, T, T), \
        f"attn weights shape mismatch: {all_attn_weights[0].shape}"

    print(f"logits shape:           {logits.shape}")
    print(f"loss:                   {loss.item():.4f}")
    print(f"attn_weights[0] shape:  {all_attn_weights[0].shape}")
    print("All assertions passed!")
