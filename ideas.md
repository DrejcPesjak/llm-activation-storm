# TODO List / Ideas

- [ ] Fix zoom activation flickering
- [ ] Add panel for logit / most probable next token - top 10
- [ ] Add button for full model architecture printout
- [ ] Check if attn mlp resid correctly extracted
- [ ] Add layer 0 (embeddings)
- [ ] Add per attn head vis
- [ ] Add more llm models and families (llama, gemma2, pythia)
- [ ] Migrate to `transformer-lens` and use `run_with_cache`
- [ ] Add activation metrics:
    - Residual anisotropy
    - Attention entropy
    - MLP activation kurtosis
    - Sink ratio
    - (also weight metrics, like QK spectral norm, etc)

## Value adds
- [ ] Add LogitLens to each layer
- [ ] Incorporate neuron descriptions
    - Transluce Llama-3.1 neuron [database](https://transluce.org/neuron-descriptions)
- [ ] Quasi circuits - select token -> highlight relevant neurons in each layer
- [ ] Metrics dashboard - plot activation metrics across layers and tokens
