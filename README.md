# LLM From Scratch

A step-by-step, lecture-style notebook series that builds a GPT-style language model from
nothing: raw text in, a trained (and fine-tuned) transformer out. No PyTorch or deep-learning
background assumed — every new tensor operation and every bit of math is introduced in plain
English the first time it's needed.

## Notebook order

| # | Notebook | Topic |
|---|----------|-------|
| 01 | `01_tokenization.ipynb` | Turning text into tokens: regex tokenization, vocabularies, special tokens |
| 02 | `02_byte_pair_encoding.ipynb` | Subword tokenization with Byte Pair Encoding (`tiktoken`) |
| 03 | `03_data_sampling.ipynb` | Sliding-window input/target pairs, PyTorch `Dataset`/`DataLoader` |
| 04 | `04_embeddings.ipynb` | Token embeddings and positional embeddings |
| 05 | `05_self_attention.ipynb` | Self-attention: from simplified to trainable (Q/K/V) |
| 05a | `05a_simplified_attention.ipynb` | *Optional deep dive:* the non-trainable mechanism step by step — dot products, softmax, context vectors |
| 05b | `05b_trainable_attention.ipynb` | *Optional deep dive:* scaled dot-product attention step by step — query/key/value matrices, `sqrt(d_k)` scaling, the `SelfAttention` module |
| 06 | `06_multihead_causal_attention.ipynb` | Causal masking, dropout, multi-head attention |
| 07 | `07_transformer_block.ipynb` | Layer norm, feed-forward networks, residual connections |
| 08 | `08_gpt_model.ipynb` | Assembling the full GPT model |
| 09 | `09_text_generation.ipynb` | Sampling text from the model (greedy, temperature, top-k) |
| 10 | `10_pretraining.ipynb` | Training the model from scratch on a small corpus |
| 11 | `11_pretrained_weights.ipynb` | Loading real pretrained GPT-2 weights |
| 12 | `12_finetuning_classification.ipynb` | Fine-tuning as a spam/ham classifier |
| 13 | `13_finetuning_instructions.ipynb` | Instruction fine-tuning |

Work through them in order — each one depends on classes/functions defined in the notebooks
before it.

## Setup

```bash
uv sync
uv run jupyter lab
```

This project follows the structure of Sebastian Raschka's *Build a Large Language Model
(From Scratch)*, reimplemented and explained from the ground up.
