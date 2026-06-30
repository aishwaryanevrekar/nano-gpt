import json
import math
import time

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate import load_model, generate_text, sample_temperature_sweep
from attention_viz import get_attention_weights, plot_attention_heads, compute_head_entropy, plot_entropy_bar

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_metrics_plot():
    try:
        data    = json.load(open('metrics.json'))
        history = data['history']
        iters  = [h['iter']       for h in history]
        trains = [h['train_loss'] for h in history]
        vals   = [h['val_loss']   for h in history]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(iters, trains, label='Train loss', color='#378ADD', linewidth=1.8)
        ax.plot(iters, vals,   label='Val loss',   color='#E24B4A', linewidth=1.8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Training curves')
        plt.tight_layout()
        cur = history[-1]
        return fig, str(cur['iter']), f"{cur['train_loss']:.4f}", f"{cur['val_loss']:.4f}"
    except Exception:
        return None, "—", "—", "—"


def generate_fn(prompt, max_tokens, temperature, top_k):
    try:
        model, encode, decode = load_model()
        start   = time.time()
        out     = generate_text(model, encode, decode, prompt,
                                int(max_tokens), float(temperature), int(top_k))
        elapsed = round((time.time() - start) * 1000)
        return out, f"{elapsed}ms"
    except FileNotFoundError:
        return "model.pt not found — run: python train.py", "—"


def sweep_fn(prompt):
    try:
        model, encode, decode = load_model()
        return sample_temperature_sweep(model, encode, decode, prompt)
    except FileNotFoundError:
        return "model.pt not found — run: python train.py"


def load_ablation():
    try:
        results = json.load(open('ablation_results.json'))
        names  = [r['name']       for r in results]
        perps  = [r['perplexity'] for r in results]
        colors = ['#4caf50', '#ff9800', '#f44336', '#795548']
        fig, ax = plt.subplots(figsize=(7, 3))
        bars = ax.barh(names, perps, color=colors[:len(names)])
        for bar, val in zip(bars, perps):
            ax.text(bar.get_width() + 0.1,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', fontsize=9)
        ax.set_xlabel('Perplexity (lower = better)')
        ax.set_title('Ablation study — contribution of each component')
        ax.invert_yaxis()
        plt.tight_layout()
        best  = min(perps)
        worst = max(perps)
        pct   = round((worst - best) / best * 100)
        md    = (f"**Best:** {names[perps.index(best)]} — perplexity {best:.1f}\n\n"
                 f"**Removing attention raises perplexity by {pct}%** vs full model.")
        return fig, md
    except Exception:
        return None, "Run `python ablation.py` first."


def visualize_fn(text, layer_idx):
    try:
        model, encode, decode = load_model()
        attn, chars  = get_attention_weights(model, encode, text, int(layer_idx))
        fig_heat     = plot_attention_heads(attn, chars, int(layer_idx))
        entropies    = compute_head_entropy(attn)
        fig_ent      = plot_entropy_bar(entropies)
        return fig_heat, fig_ent
    except FileNotFoundError:
        return None, None


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

with gr.Blocks(title="Nano-GPT") as demo:

    gr.Markdown("# Nano-GPT — Transformer from Scratch")
    gr.Markdown(
        "Character-level GPT trained on Shakespeare · "
        "Built from scratch in PyTorch · "
        "Dr. Mahapatra | Intro to AI"
    )

    # ── Tab 1: Training ────────────────────────────────────────────────────
    with gr.Tab("Training"):
        gr.Markdown("### Training curves — auto-refreshes every 5s while train.py is running")
        with gr.Row():
            iter_box  = gr.Textbox(label="Iteration",  interactive=False, scale=1)
            train_box = gr.Textbox(label="Train loss", interactive=False, scale=1)
            val_box   = gr.Textbox(label="Val loss",   interactive=False, scale=1)
        loss_plot   = gr.Plot(label="Loss curves")
        refresh_btn = gr.Button("Refresh now")
        timer = gr.Timer(value=5)
        timer.tick(fn=load_metrics_plot,
                   outputs=[loss_plot, iter_box, train_box, val_box])
        refresh_btn.click(fn=load_metrics_plot,
                          outputs=[loss_plot, iter_box, train_box, val_box])
        demo.load(fn=load_metrics_plot,
                  outputs=[loss_plot, iter_box, train_box, val_box])

    # ── Tab 2: Generate ────────────────────────────────────────────────────
    with gr.Tab("Generate"):
        gr.Markdown("### Text generation")
        with gr.Row():
            with gr.Column(scale=1):
                prompt_in = gr.Textbox(label="Prompt", value="HAMLET:\n", lines=3)
                max_tok   = gr.Slider(50, 500, value=200, step=10,  label="Max tokens")
                temp_sl   = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                topk_sl   = gr.Slider(1, 100,   value=40,  step=1,   label="Top-k")
                gen_btn   = gr.Button("Generate", variant="primary")
                time_box  = gr.Textbox(label="Generation time", interactive=False)
            with gr.Column(scale=1):
                gen_out = gr.Textbox(label="Generated text", lines=15, interactive=False)
        gen_btn.click(fn=generate_fn,
                      inputs=[prompt_in, max_tok, temp_sl, topk_sl],
                      outputs=[gen_out, time_box])
        gr.Markdown("#### Temperature sweep — same prompt at 0.2 / 0.8 / 1.5")
        sweep_btn = gr.Button("Run sweep")
        sweep_out = gr.Textbox(label="Sweep output", lines=12, interactive=False)
        sweep_btn.click(fn=sweep_fn, inputs=prompt_in, outputs=sweep_out)

    # ── Tab 3: Ablation ────────────────────────────────────────────────────
    with gr.Tab("Ablation"):
        gr.Markdown("### Ablation study — proof that every component earns its place")
        gr.Markdown(
            "Run `python ablation.py` first (~20 min), "
            "or `python ablation.py --quick` for a test run."
        )
        abl_btn  = gr.Button("Load results")
        abl_plot = gr.Plot()
        abl_md   = gr.Markdown()
        abl_btn.click(fn=load_ablation, outputs=[abl_plot, abl_md])
        demo.load(fn=load_ablation,    outputs=[abl_plot, abl_md])

    # ── Tab 4: Attention ───────────────────────────────────────────────────
    with gr.Tab("Attention"):
        gr.Markdown("### Attention head visualizer — see inside the transformer")
        with gr.Row():
            attn_text = gr.Textbox(label="Input text",
                                   value="To be or not to be", scale=3)
            layer_sl  = gr.Slider(0, 3, value=0, step=1, label="Layer", scale=1)
        viz_btn = gr.Button("Visualize")
        with gr.Row():
            heat_plot = gr.Plot(label="Attention heatmaps — all 4 heads")
            ent_plot  = gr.Plot(label="Head entropy — lower = more focused")
        viz_btn.click(fn=visualize_fn,
                      inputs=[attn_text, layer_sl],
                      outputs=[heat_plot, ent_plot])
        gr.Markdown(
            "*Low entropy = head has specialized in specific token relationships. "
            "High entropy = diffuse, early-layer behaviour.*"
        )

demo.launch(server_name="0.0.0.0", theme=gr.themes.Soft())
