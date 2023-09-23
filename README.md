# WIP: gpt-neox + llama

Forked from GPT-NeoX.

## Quick Start

1. Prepare data
   ```bash
   python tools/preprocess_data.py --input <JSONL_FILE> --tokenizer-type HFAutoTokenizer --vocab-file <HF_TOKENIZER_NAME_OR_PATH> --append-eod --output-prefix <OUTPUT_MMAP_FILE_PATH_PREFIX>
   ```
   which preprocesses jsonl data to tokenized mmap data

2. Prepare model
   ```bash
   python tools/convert_llama_from_hf.py
   ```
   And please edit the path in the python script.

3. Define ds configs
   Examples can be found in `configs/nebula`

4. Run

   Example:
   ```bash
   python deepy.py train.py configs/nebula/llama_data.yml configs/nebula/7B_llama_baseline.yml
   ```

6. Convert checkpoint to HF

   TODO

## Common Issues

```
ImportError: cannot import name 'helpers' from 'megatron.data' 
```

`cd ./megatron/data && make` (see: https://github.com/EleutherAI/gpt-neox/issues/934).
