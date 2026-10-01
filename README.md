# Transformer Visualizer

A tiny GPT, written from scratch in plain JavaScript, that you train **live in your browser** while watching what it does to tokens, attention and predictions. Every knob of a GPT-style model is exposed, and each change shows up immediately.

No install, no build, no server, no network calls. Just open `index.html`.

## Run it

```bash
git clone <this repo>
open index.html          # macOS. On Windows/Linux, double-click the file.
```

To run the tests (Node 18+, no dependencies):

```bash
npm test
```

## What you can turn

| Group | Knob | What to watch |
|---|---|---|
| **Data & tokenizer** | Training text (patterns / refinery ops / Shakespeare / your own) | Loss curve, generated text style |
| | Tokenizer: character, BPE, word | Token chips, chars/token, `<unk>` count |
| | Vocab size (BPE/word) | The BPE merge list grows; prompts take fewer tokens; embedding params grow |
| **Architecture** *(resets weights)* | Layers, d_model, heads, context length, MLP expansion | Pipeline diagram, parameter breakdown, FLOPs, number of attention maps |
| | Positional encoding: learned / sinusoidal / none | With *none*, the model can't tell word order — loss plateaus higher |
| | Tie input/output embeddings | Parameter count drops by V × d |
| | Init seed | Different random starting point |
| **Training** | Learning rate, batch size | Loss curve speed and noise; too high → NaN (it tells you) |
| | Causal mask while training | Turn it off and the model "cheats" by looking at the answer: train loss collapses, generation is garbage |
| **Inference** *(live, no retrain)* | Temperature, top-k, top-p | Next-token bars (filled = after knobs, outline = raw), entropy, generated text |
| | Attention sharpness (scale multiplier on 1/√d_head) | Heatmaps go from uniform (0×) to one-hot (4×) |
| | Causal mask at inference | Upper triangle of the heatmaps lights up; predictions break |

## Views

- **Architecture** — the GPT-2 block layout with live tensor shapes, parameter breakdown by component, and a model card (FLOPs, memory, tokens-per-parameter vs. GPT-2 small).
- **Tokenizer** — your prompt split into tokens with their ids. Click a token to see where it looks (underline strength = attention weight).
- **Training** — train loss per step and held-out validation loss every 25 steps, against the "uniform guess" baseline ln(V). Flags overfitting when it happens.
- **Attention** — heatmap for any layer/head plus thumbnails of all of them, with head statistics (entropy, previous-token, self, first-token) and a plain-English guess at what the head does.
- **Next-token prediction** — top-15 candidates and an autoregressive **Generate** button.
- **Residual stream** — how much each block writes into the residual stream.

## Suggested experiments (5 minutes each)

1. **Watch structure appear.** Select *Patterns*, press Train, and watch the attention thumbnails go from grey mush to sharp diagonals (previous-token heads) within ~100 steps.
2. **Tokenizer trade-off.** Same text with *Character* vs *BPE 512*. Character: tiny vocab, long sequences. BPE: fewer tokens, bigger embedding table.
3. **Cheating without a mask.** Uncheck *Causal mask while training*, train. Loss drops to near 0 — then generate and see that the model learned nothing useful.
4. **Temperature.** On a trained model, slide temperature 0 → 2 and read the entropy numbers.
5. **Overfitting.** Train on *Refinery operations* for a few hundred steps: train loss goes under 1, validation loss climbs past the uniform baseline. That's what a 30k-parameter model does with 700 tokens.

## Honest limits

- It's ~10k–200k parameters trained on a few KB of text. It learns spelling, common phrases and patterns, not meaning.
- Everything runs on the main thread in plain JS (~10–20 steps/s for the default size). Big settings (6 layers × d=64 × context 96) are slow.
- BPE here works on characters, not bytes, so characters never seen in training become `<unk>` (shown in red).

## Code layout

```
index.html          page + controls
css/style.css       light/dark theme
js/tensor.js        2-D tensors + reverse-mode autograd (matmul, LayerNorm, GELU, fused causal MHA, cross-entropy)
js/tokenizer.js     char / word / BPE tokenizers
js/model.js         GPT (pre-LN blocks, learned/sinusoidal positions, weight tying), Adam, sampling
js/corpora.js       built-in training texts
js/charts.js        canvas charts (loss, heatmaps, probability bars)
js/app.js           UI wiring
tests/              node:test — finite-difference gradient checks, tokenizer round-trips, causality, training sanity
```
