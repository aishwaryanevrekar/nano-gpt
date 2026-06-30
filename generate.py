import argparse
import json
import pathlib
import torch
from model import GPT, BLOCK_SIZE

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def load_model(ckpt='model.pt', vocab='vocab.json'):
    with open(vocab, 'r') as f:
        v = json.load(f)

    stoi = v['stoi']
    itos = {int(k): ch for k, ch in v['itos'].items()}
    vocab_size = v['vocab_size']

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return ''.join(itos[i] for i in ids)

    checkpoint = torch.load(ckpt, map_location=DEVICE)
    model = GPT(checkpoint['vocab_size'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.to(DEVICE)

    return model, encode, decode


def generate_text(model, encode, decode,
                  prompt='', max_tokens=200, temperature=0.8, top_k=40):
    if prompt:
        ids = encode(prompt)
    else:
        ids = [0]

    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    out = model.generate(idx, max_new_tokens=max_tokens,
                         temperature=temperature, top_k=top_k)
    return decode(out[0].tolist())


def sample_temperature_sweep(model, encode, decode, prompt,
                              temps=None):
    if temps is None:
        temps = [0.2, 0.8, 1.5]

    label_map = {0.2: 'focused', 0.8: 'balanced', 1.5: 'creative'}
    sections = []
    for t in temps:
        label = label_map.get(t, str(t))
        text  = generate_text(model, encode, decode,
                               prompt=prompt, max_tokens=100,
                               temperature=t, top_k=40)
        header = f"── temperature={t} ({label}) {'─' * (44 - len(str(t)))}"
        sections.append(f"{header}\n{text}")

    return '\n\n'.join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text with nano-GPT')
    parser.add_argument('--prompt',      type=str,   default='HAMLET:\n')
    parser.add_argument('--tokens',      type=int,   default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k',       type=int,   default=40)
    args = parser.parse_args()

    if not pathlib.Path('model.pt').exists():
        print("Run python train.py first")
    else:
        model, encode, decode = load_model()
        output = generate_text(model, encode, decode,
                               prompt=args.prompt,
                               max_tokens=args.tokens,
                               temperature=args.temperature,
                               top_k=args.top_k)
        print(output)
