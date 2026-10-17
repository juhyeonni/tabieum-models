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
- **Output:** `embedding`, shape `(1, 384)`, float16, an L2-normalized vector. Mean pooling and normalization are inside the model.
- **Prefixes:** as for the original model, start queries with `query: ` and documents with `passage: `.
- **Requires:** macOS 15 or iOS 18 or later. Run it with `MLComputeUnits.cpuAndNeuralEngine`. Avoid `.all`: it was slower in our test (3.9 ms against 2.2 ms at length 128).
- **Size:** 224 MB (`e5small-fp16-256.mlpackage.zip`). Verify it with `SHA256SUMS`.

## Why length 256, and why a fixed length

The Neural Engine needs static shapes. A model converted with `EnumeratedShapes` (64/128/256) ran at CPU speed in our test, with no Neural Engine memory use, so this repository ships one fixed-length model. 256 tokens covers a 300-character chunk plus a heading path in Korean (up to 208 tokens in our tests), Japanese (161) and English (123).

## Measurements

<!-- results:begin speed -->
Apple M1 Pro, macOS 26.5.2, one text per call, Core ML via Swift. These are our own measurements, not a standard benchmark.

| Compute units | ms per text (length 256) |
|---|---|
| `cpuOnly` | 12.0 |
| `cpuAndNeuralEngine` | 6.0 |

- **Memory** (`footprint` of the process, ANE included): about 65 MB (20 MB in the app, 45 MB owned by the Neural Engine). The first load compiles for the Neural Engine and takes a few seconds.
- **Fidelity:** cosine similarity to the PyTorch model on the same text is 0.99995 on the Neural Engine and 1.00000 on CPU.
<!-- results:end speed -->

### Retrieval quality (reference only)

<!-- results:begin retrieval -->
Hybrid search (SQLite FTS5 keyword ranking fused with these vectors by reciprocal rank fusion), natural-language queries. The notes and queries are synthetic and AI-written, so compare models against each other rather than reading absolute numbers.

| Set | Queries | R@1 % (95% CI) | R@3 % (95% CI) | R@10 % (95% CI) |
|---|---|---|---|---|
| English | 120 | 65.8 (57–74) | 85.0 (78–90) | 96.7 (92–99) |
| Korean | 120 | 41.7 (33–51) | 69.2 (60–77) | 84.2 (77–90) |
| Mixed Korean/English | 93 | 41.9 (32–52) | 58.1 (48–68) | 78.5 (69–86) |

Recall at 3 (%) against other options on the same queries. Keyword search requires every query word to appear, so it finds nothing for these paraphrased queries.

| Option | English | Korean | Mixed Korean/English |
|---|---|---|---|
| This model (hybrid search) | 85.0 | 69.2 | 58.1 |
| Keyword search alone (tabieum FTS5) | 0.0 | 0.0 | 0.0 |
| potion-multilingual-128M (static, not shipped) | 57.5 | 54.2 | 34.4 |
| e5-small 8-bit palettized (not shipped) | 85.0 | 70.0 | 58.1 |
| e5-small 6-bit palettized (not shipped) | 85.8 | 67.5 | 60.2 |
| e5-small 4-bit palettized (not shipped) | 83.3 | 59.2 | 49.5 |
<!-- results:end retrieval -->

The Core ML model reproduces the scores of the original PyTorch model. The static-embedding model and the palettized variants above are not offered: the static model scores far lower, and 4-bit palettization drops Korean and mixed text.

## Palettized variants

Per-tensor palettization shrinks the file (8-bit 112 MB, 6-bit 84 MB, 4-bit 56 MB) with the same quality down to 6-bit, but Core ML then decompresses the weights into the app's memory (about 570 MB in our test) and the first load takes about 8 seconds. This repository therefore ships fp16 only.

## Reproduce the conversion

The tooling is in [juhyeonni/tabieum-models](https://github.com/juhyeonni/tabieum-models). From a clone of it:

```
uv sync --project e5-small-coreml
uv run --project e5-small-coreml python e5-small-coreml/convert.py 256
```

`e5-small-coreml/pyproject.toml` pins the versions we used (`uv.lock` has the full set). `coremltools` 9.0 is validated up to PyTorch 2.7; newer PyTorch failed at conversion. `convert.py` replaces the attention-mask fill value with `-1e4`, because the default (-inf in fp16) made CPU and GPU return NaN.

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
