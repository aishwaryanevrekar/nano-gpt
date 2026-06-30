import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import BLOCK_SIZE


def get_attention_weights(model, encode, text, layer_idx=0):
    ids  = encode(text)[:BLOCK_SIZE]
    chars = list(text[:len(ids)])
    idx  = torch.tensor([ids], dtype=torch.long,
                        device=next(model.parameters()).device)
    with torch.no_grad():
        _, _, all_attn = model(idx)
    attn = all_attn[layer_idx][0].detach().cpu().numpy()  # (n_heads, T, T)
    return attn, chars


def plot_attention_heads(attn, chars, layer_idx):
    n_heads, T, _ = attn.shape
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(attn[h], cmap='Blues', aspect='auto', vmin=0)
        ax.set_title(f'Head {h + 1}', fontsize=11)
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(chars, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(chars, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'Layer {layer_idx} attention', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def compute_head_entropy(attn):
    avg_attn = attn.mean(axis=1) + 1e-9          # (n_heads, T)
    entropy  = -(avg_attn * np.log(avg_attn)).sum(axis=-1)  # (n_heads,)
    return entropy


def plot_entropy_bar(entropies):
    n_heads = len(entropies)
    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(range(n_heads), entropies, color='steelblue', width=0.6)
    for bar, val in zip(bars, entropies):
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f'Head {i + 1}' for i in range(n_heads)])
    ax.set_ylabel('Entropy (lower = more focused)')
    ax.set_title('Head specialization across attention heads')
    ax.set_ylim(0, max(entropies) * 1.2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# __main__: demo when model.pt is available
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if not pathlib.Path('model.pt').exists():
        print("Run python train.py first")
    else:
        from generate import load_model

        model, encode, decode = load_model()
        prompt = 'HAMLET:\nTo be, or not to be'

        for layer_idx in range(2):
            attn, chars = get_attention_weights(model, encode, prompt, layer_idx=layer_idx)

            fig_heads = plot_attention_heads(attn, chars, layer_idx)
            out_heads = f'attn_layer{layer_idx}_heads.png'
            fig_heads.savefig(out_heads, dpi=150, bbox_inches='tight')
            plt.close(fig_heads)
            print(f'Saved {out_heads}')

            entropies = compute_head_entropy(attn)
            print(f'Layer {layer_idx} head entropies: {np.round(entropies, 3)}')

            fig_ent = plot_entropy_bar(entropies)
            out_ent = f'entropy_layer{layer_idx}.png'
            fig_ent.savefig(out_ent, dpi=150, bbox_inches='tight')
            plt.close(fig_ent)
            print(f'Saved {out_ent}')
