---
license: mit
base_model: intfloat/multilingual-e5-small
library_name: coreml
pipeline_tag: feature-extraction
language:
- multilingual
- ko
- en
- ja
tags:
- coreml
- apple-silicon
- sentence-embeddings
- e5
---

# multilingual-e5-small, Core ML (fp16, sequence length 256)

[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) converted to a Core ML `mlprogram` for Apple silicon. It is the optional semantic-search model of [tabieum](https://github.com/juhyeonni/tabieum), and it works on its own.

- **Input:** `input_ids` and `attention_mask`, both `int32`, shape `(1, 256)`. Tokenize with the original XLM-R tokenizer (`intfloat/multilingual-e5-small`), pad to 256, truncate longer text.
- **Output:** `embedding`, a 384-d L2-normalized vector. Mean pooling and normalization are inside the model.
- **Prefixes:** as for the original model, start queries with `query: ` and documents with `passage: `.
- **Requires:** macOS 15 or iOS 18 or later. Run it with `MLComputeUnits.cpuAndNeuralEngine`. Avoid `.all`: it was slower in our test (3.9 ms against 2.2 ms at length 128).
- **Size:** 224 MB (`e5small-fp16-256.mlpackage.zip`). Verify it with `SHA256SUMS`.

## Why length 256, and why a fixed length

The Neural Engine needs static shapes. A model converted with `EnumeratedShapes` (64/128/256) ran at CPU speed in our test, with no Neural Engine memory use, so this repository ships one fixed-length model. 256 tokens covers a 300-character chunk plus a heading path in Korean (up to 208 tokens in our tests), Japanese (161) and English (123).

## Measurements

Apple M1 Pro, macOS 26.5.2, one text per call, Core ML via Swift. These are our own measurements, not a standard benchmark.

| Compute units | ms per text (length 256) |
|---|---|
| `cpuOnly` | 12.0 |
| `cpuAndNeuralEngine` | 6.0 |

- **Memory** (`footprint` of the process, ANE included): about 65 MB (20 MB in the app, 45 MB owned by the Neural Engine). The first load compiles for the Neural Engine and takes a few seconds.
- **Fidelity:** cosine similarity to the PyTorch model on the same text is 0.99995 on the Neural Engine and 1.00000 on CPU and GPU.

### Retrieval quality (reference only)

Hybrid search (SQLite FTS5 keyword ranking fused with these vectors by reciprocal rank fusion), natural-language queries, recall at 3, in percent. The notes and queries are synthetic, AI-written, so compare models against each other rather than reading absolute numbers.

| Set | Queries | R@3 |
|---|---|---|
| English | 120 | 85.0 |
| Korean | 120 | 69.2 |
| Mixed Korean/English | 93 | 58.1 |

The Core ML model reproduces the scores of the original PyTorch model. In the same test, a static-embedding model (potion-multilingual-128M) scored 57.5 / 54.2 / 34.4, and 4-bit palettization of this model dropped Korean to 59.2, so neither is offered here.

## Palettized variants

Per-tensor palettization shrinks the file (8-bit 112 MB, 6-bit 84 MB, 4-bit 56 MB) with the same quality down to 6-bit, but Core ML then decompresses the weights into the app's memory (about 570 MB in our test) and the first load takes about 8 seconds. This repository therefore ships fp16 only.

## Reproduce the conversion

```
pip install -r requirements.txt
python convert.py 256
```

`requirements.txt` pins the versions we used. `coremltools` 9.0 is validated up to PyTorch 2.7; newer PyTorch failed at conversion. `convert.py` replaces the attention-mask fill value with `-1e4`, because the default (-inf in fp16) made CPU and GPU return NaN.

## License and credits

The weights are derived from `intfloat/multilingual-e5-small`, which is published under the MIT License. This conversion is released under the same MIT License. Please cite the original work:

```
@article{wang2024multilingual,
  title={Multilingual E5 Text Embeddings: A Technical Report},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Yang, Linjun and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2402.05672},
  year={2024}
}
```
