# tabieum-models

Conversion, evaluation and release tooling for the optional embedding models of [tabieum](https://github.com/juhyeonni/tabieum). The weights live on Hugging Face, one repository per model; this repository holds how they are made and measured.

| Model | Hugging Face | Format |
|---|---|---|
| `e5-small-coreml/` | [juhyeonni/tabieum-e5-small-coreml](https://huggingface.co/juhyeonni/tabieum-e5-small-coreml) | Core ML fp16, fixed length 256, from [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) |

## Layout

- `<model>/convert.py`, `<model>/pyproject.toml`: the conversion, with pinned tool versions (`uv.lock` at the root pins the full set).
- `<model>/MODEL_CARD.md`: the card uploaded as the Hugging Face `README.md`. The speed, memory and retrieval tables between `results:begin/end` markers are generated from the results file; do not edit them by hand.
- `eval/`: retrieval evaluation on synthetic notes (English, Korean, mixed). `data/*.json` are the notes and queries; `chunks-*.json` are the texts as tabieum's chunker produces them (with `passage: ` and `query: ` prefixes); `fts.json` is the keyword ranking of tabieum's search for each query, used for hybrid scoring. The notes and queries are AI-written, so scores compare models with each other and are not absolute.
- `eval/results/<model>/<version>.json`: the recorded measurements of one release (package sha256, tool versions, chip and macOS, dataset hash, hits per set, comparison scores, speed, memory).
- `bench/`: a Swift program that loads a package with chosen compute units and reports load time, milliseconds per text and memory.
- `scripts/check.sh`: the checks below.

## Convert, evaluate, measure

```
uv sync --project e5-small-coreml
uv run --project e5-small-coreml python e5-small-coreml/convert.py 256   # writes e5small_fp16_256.mlpackage
uv run --project eval python eval/run_eval.py e5small_fp16_256.mlpackage --length 256 \
    --zip e5small-fp16-256.mlpackage.zip --out eval/results/e5-small-coreml/fp16-256.json
python3 eval/make_card.py                                                # re-render the card tables
cd bench && swift build -c release
PKG=../e5small_fp16_256.mlpackage S=256 UNITS=ne .build/release/cml     # UNITS: ne, cpu, gpu, all
footprint <pid>    # while the program holds the model (set HOLD=seconds): shows the Neural Engine share
```

`run_eval.py` rewrites the measured fields of the results file and keeps the rest. Speed, memory, fidelity and the comparison scores of other options are recorded by hand from the Swift bench and earlier runs.

Memory note: Activity Monitor omits the Neural Engine share ("Owned physical footprint (neural)"). Measure with `footprint`.

## Checks

```
scripts/check.sh                       # stage 1
scripts/check.sh e5small-fp16-256.mlpackage.zip   # stages 1 and 2 (also accepts a .mlpackage)
```

Stage 1 needs no model: ruff, `uv lock --check`, the results files (schema, recall ordering, dataset hash, package sha256), the card front matter and generated tables, and the bench build. Stage 2 takes a package: input and output contract (`input_ids` and `attention_mask` int32 `(1, 256)`, `embedding` `(1, 384)` float16), cosine of at least 0.999 against the PyTorch model on CPU, no NaN on any compute unit, and, for a zip, its sha256 against the results file and `SHA256SUMS`. Speed, memory and Neural Engine quality are measured by hand on a real Mac.

## Release a model

```
ditto -c -k --sequesterRsrc --keepParent e5small_fp16_256.mlpackage e5small-fp16-256.mlpackage.zip
shasum -a 256 e5small-fp16-256.mlpackage.zip > SHA256SUMS
# stage the zip, SHA256SUMS, convert.py, pyproject.toml and MODEL_CARD.md (as README.md) in one folder, then:
hf upload juhyeonni/tabieum-e5-small-coreml <folder> .
```

The app downloads a pinned commit, so a new upload never changes an installed app.

## License

Apache-2.0 for the code in this repository (see `LICENSE`). The converted weights are derived from `intfloat/multilingual-e5-small` and are released under the MIT License; see each model card.
