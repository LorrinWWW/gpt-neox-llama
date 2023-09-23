# Savanna

Transformer alternatives (pretraining, evals, inference, synthetics). Convergence repository that will contain safari-neox and bigsafari utilities.

Forked from GPT-NeoX.

## Environment Variables

Set `SAVANNA_PATH` to the main folder.

## Common Issues

```
ImportError: cannot import name 'helpers' from 'megatron.data' 
```

`cd ./megatron/data && make` (see: https://github.com/EleutherAI/gpt-neox/issues/934).