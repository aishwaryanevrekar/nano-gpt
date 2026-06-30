
# Nano-GPT — Transformer from Scratch

A character-level GPT model built entirely from scratch in PyTorch, trained on the complete works of Shakespeare. This project implements a decoder-only transformer architecture with multi-head self-attention, demonstrating the core ideas behind large language models like GPT in a minimal and educational codebase.

Built as part of **Introduction to Artificial Intelligence** course under **Dr. Mahapatra** at **Jio Institute**, Term 2.

---

## Team Members

| Name | GitHub |
|---|---|
| Aishwarya Nevrekar | [@aishwaryanevrekar](https://github.com/aishwaryanevrekar) |
| Kushagra Gupta | [@kushagragupta-23](https://github.com/kushagragupta-23) |

---

## What This Project Does

This project implements a miniature GPT from the ground up — no HuggingFace, no pre-trained weights — to demonstrate how transformers actually work internally. The interactive Gradio app lets you:

- **Generate text** in the style of Shakespeare with adjustable temperature and top-k sampling
- **Visualize training curves** (train loss vs. validation loss over iterations)
- **Explore attention heads** — see exactly which tokens each head attends to
- **Run ablation studies** — quantitatively measure the contribution of each component (attention, feedforward, positional encoding)

---

## Model Architecture

| Hyperparameter | Value |
|---|---|
| Layers | 4 transformer blocks |
| Attention heads | 4 heads per layer |
| Embedding dimension | 128 |
| Context window | 64 tokens |
| Dropout | 0.1 |
| Parameters | ~800K |
| Training data | Shakespeare (1.1M characters) |

The model uses:
- Token + positional embeddings
- Multi-head causal self-attention with masking
- Layer normalization (pre-norm style)
- GELU feedforward networks
- Cross-entropy language modelling loss

---

## Project Structure

```
nano-gpt/
├── app.py              # Gradio web interface (4 tabs)
├── model.py            # GPT architecture from scratch
├── train.py            # Training loop with checkpointing
├── generate.py         # Text generation + temperature sweep
├── attention_viz.py    # Attention weight extraction and plotting
├── ablation.py         # Ablation study across model variants
├── model.pt            # Trained model checkpoint
├── vocab.json          # Character vocabulary (encode/decode)
├── metrics.json        # Training history (loss per iteration)
└── requirements.txt    # Python dependencies
```

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/aishwaryanevrekar/nano-gpt.git
cd nano-gpt
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. (Optional) Train the model yourself

A pre-trained `model.pt` is already included. If you want to retrain from scratch:

```bash
python train.py
```

Training takes approximately 5–10 minutes on a modern CPU, or under 2 minutes on a GPU/Apple Silicon (MPS). Loss is logged to `metrics.json` every 100 iterations.

### 4. Launch the app

```bash
python app.py
```

Open your browser at `http://localhost:7860`

### 5. (Optional) Run the ablation study

```bash
python ablation.py          # Full study (~20 min)
python ablation.py --quick  # Quick test run (~2 min)
```

Results are saved to `ablation_results.json` and appear in the Ablation tab of the app.

---

## App Tabs

### Training
Live training curves showing train loss and validation loss over iterations. Auto-refreshes every 5 seconds while `train.py` is running.

### Generate
Enter any Shakespeare-style prompt and generate text. Controls:
- **Max tokens** — how many characters to generate (50–500)
- **Temperature** — higher = more creative/random, lower = more focused
- **Top-k** — limits sampling to the top-k most likely next tokens
- **Temperature sweep** — generates the same prompt at three temperatures (0.2 / 0.8 / 1.5) side by side

### Ablation
Bar chart comparing perplexity across four model variants:
- Full model
- No attention (feedforward only)
- No positional encoding
- No feedforward network

### Attention
Heatmap visualization of all 4 attention heads for any input text, plus a head entropy bar chart showing which heads are specialized (low entropy) vs. diffuse (high entropy).

---

## Key Results

- Final train loss: **1.76** | Validation loss: **1.85**
- Removing multi-head attention raises perplexity significantly, confirming its central role
- Individual attention heads develop distinct specializations across layers

---

## Dependencies

- `torch` — model training and inference
- `gradio` — interactive web interface
- `matplotlib` — training curves and attention plots
- `numpy` — numerical operations
