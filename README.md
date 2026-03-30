# Activation Storm

![Activation Storm screenshot](img/screenshot_2026-03-30w.webp)

Activation Storm is a local browser app for visualizing activations from a prompt as they move through an LLM layer by layer.

You enter a prompt, run one forward pass, and inspect the flow across the network. The top band shows the full model at once. The lower panel shows the currently selected layer-stage slice as a `tokens x hidden_dim` field. The timeline moves through layer computation steps, not generated output tokens.

The app now supports:
- custom `google/gemma-3-4b-it` handling for the low-VRAM Gemma 3 path
- a small TransformerLens-backed model set from the dropdown, including GPT-2, Pythia, Llama, Mistral, Qwen, Gemma 2, and Gemma 3 1B variants
    - can expand to all **237 models** in the `transformer-lens` zoo (add string identifiers to `TL_MODEL_SPECS` in `src/activation_storm/adapters.py`)

For each layer, the app captures:
- `attn_out`
- `resid_after_attn`
- `mlp_out`
- `resid_after_mlp`

The UI supports:
- filtering by `Attn`, `Resid`, and `MLP`
- optionally including special/chat-wrapper tokens in the token axis
- stepping or playing through the captured layer-stage sequence

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch accelerate transformer-lens
```

Run:

```bash
python -m src.activation_storm --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

Notes:
- this visualizes activations only, not weights
- practical prompt length is still limited by model context and available memory
