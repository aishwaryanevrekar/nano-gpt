import argparse
import json
import torch
import torch.nn as nn
from model import GPT, BLOCK_SIZE, count_params

MAX_ITERS  = 3000
EVAL_EVERY = 100
EVAL_ITERS = 50
LR         = 3e-4
BATCH_SIZE = 32

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars     = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return ''.join(itos[i] for i in ids)

with open('vocab.json', 'w') as f:
    json.dump({'stoi': stoi, 'itos': {str(k): v for k, v in itos.items()},
               'vocab_size': vocab_size}, f)

data   = torch.tensor(encode(text), dtype=torch.long)
split  = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]

# ---------------------------------------------------------------------------
# Batch sampling
# ---------------------------------------------------------------------------

def get_batch(split):
    src = train_data if split == 'train' else val_data
    ix  = torch.randint(len(src) - BLOCK_SIZE, (BATCH_SIZE,))
    x   = torch.stack([src[i:i + BLOCK_SIZE]     for i in ix])
    y   = torch.stack([src[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ---------------------------------------------------------------------------
# Loss estimation
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

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(max_iters, eval_every):
    model = GPT(vocab_size).to(DEVICE)
    print(f"Device: {DEVICE}  |  Parameters: {count_params(model)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    history   = []

    for step in range(max_iters):
        # Periodic evaluation
        if step % eval_every == 0 or step == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
            entry = {'iter': step, 'train_loss': losses['train'], 'val_loss': losses['val']}
            history.append(entry)
            with open('metrics.json', 'w') as f:
                json.dump({'history': history, 'current': entry}, f, indent=2)

        # Forward + backward
        x, y = get_batch('train')
        _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save({'model_state': model.state_dict(), 'vocab_size': vocab_size}, 'model.pt')
    print(f"\nTraining complete. Model saved to model.pt")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train nano-GPT on Shakespeare')
    parser.add_argument('--iters', type=int, default=MAX_ITERS,
                        help=f'Number of training iterations (default: {MAX_ITERS})')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run: 300 iters, eval every 50')
    args = parser.parse_args()

    max_iters  = 300        if args.quick else args.iters
    eval_every = 50         if args.quick else EVAL_EVERY

    train(max_iters, eval_every)
