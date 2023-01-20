# How to Pretrain UCTopic

## 0. Prepare your environment
`pip install -r requierements_ja.txt`

### Caution
If you want to pretrain the UCTopic based on JapaneseBERT, you use [this modified transformers](https://github.com/Katsumata420/transformers/tree/luke-japanese-tokenizer).

```bash
$ pip uninstall transformers
$ mkdir tools && cd tools && git clone https://github.com/Katsumata420/transformers.git && cd transformers
$ git fetch upstream luke-japanese-tokenizer:luke-japanese-tokenizer && git checkout luke-japanese-tokenizer
$ pip install -e .
```

## 1. Prepare Training data
Prepare training data using Luke scripts.

Please see [the forked Luke repository](https://github.com/Katsumata420/luke/tree/pretraining-dataset-jsonl).

Pretraining data format is following:

```json
{"text": ..., "selected_entities": [[entity_title, entity_id, start, end], ...]}
```

## 2. Run pretrain.py

See `run_japanese.sh`.

# Option
- If you pretrain the UCTopic from Luke based on JapaneseBERT, you need to modify the `pretrain.py`. 
  - You need to use LukeJapaneseTokenizer instead of AutoTokenizer.
