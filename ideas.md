# TODO List / Ideas

- [x] Fix zoom activation flickering (not a bug, just too many dimensions and too little pixels)
- [ ] Move the layer slider and play/pause button to bottom of screen, always visible floating
- [x] Add panel for logit / most probable next token - top 10
- [x] Add button for full model architecture printout (click model, window popup, `print(model)`)
- [x] Check if attn mlp resid correctly extracted (`debug/extraction-check`)
- [x] Add layer 0 (embeddings)
- [ ] Add per attn head vis
- [ ] Dynamic UI layout (Bootstrap or Tailwind, right now too many hardcoded CSS px values)
- [x] Add more llm models and families (llama, gemma2, pythia, qwen, mistral, etc)
- [x] Migrate to `transformer-lens` and use `run_with_cache`
    - [x] TransformerLens adapter
    - [x] Qwen loading problems (transformers 5.4 incompatibility)
    - [ ] Might have a memory leak (VRAM model switching is ok)
- [ ] Add activation metrics:
    - Residual anisotropy
    - Attention entropy
    - MLP activation kurtosis
    - Sink ratio
    - (also weight metrics, like QK spectral norm, etc)

## Value adds
- [x] Add LogitLens to each layer (or Patchscopes)
- [ ] Incorporate neuron descriptions
    - Transluce Llama-3.1 neuron [database](https://transluce.org/neuron-descriptions)
- [ ] Quasi circuits - select token -> highlight relevant neurons in each layer
- [ ] Metrics dashboard - plot activation metrics across layers and tokens
