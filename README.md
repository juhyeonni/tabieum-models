# tabieum-models

Conversion, evaluation and release tooling for the optional embedding models of [tabieum](https://github.com/juhyeonni/tabieum). The weights live on Hugging Face, one repository per model; this repository holds how they are made and measured.

| Model | Hugging Face | Format |
|---|---|---|
| `e5-small-coreml/` | [juhyeonni/tabieum-e5-small-coreml](https://huggingface.co/juhyeonni/tabieum-e5-small-coreml) | Core ML fp16, fixed length 256, from [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) |

## Layout

- `<model>/convert.py`, `requirements.txt`: the conversion, with pinned tool versions.
- `<model>/MODEL_CARD.md`: the card uploaded as the Hugging Face `README.md`.
- `eval/`: retrieval evaluation on synthetic notes (English, Korean, mixed). `data/*.json` are the notes and queries; `chunks-*.json` are the texts as tabieum's chunker produces them (with `passage: ` and `query: ` prefixes); `fts.json` is the keyword ranking of tabieum's search for each query, used for hybrid scoring. The notes and queries are AI-written, so scores compare models with each other and are not absolute.
- `bench/`: a Swift program that loads a package with chosen compute units and reports load time, milliseconds per text and memory.

## Convert, evaluate, measure

```
python -m venv .venv && . .venv/bin/activate
pip install -r e5-small-coreml/requirements.txt
python e5-small-coreml/convert.py 256            # writes e5small_fp16_256.mlpackage
python eval/eval_coreml.py e5small_fp16_256.mlpackage 256
cd bench && swift build -c release
PKG=../e5small_fp16_256.mlpackage S=256 UNITS=ne .build/release/cml   # UNITS: ne, cpu, gpu, all
footprint <pid>    # while the program holds the model (set HOLD=seconds): shows the Neural Engine share
```

Memory note: Activity Monitor omits the Neural Engine share ("Owned physical footprint (neural)"). Measure with `footprint`.

## Release a model

```
ditto -c -k --sequesterRsrc --keepParent e5small_fp16_256.mlpackage e5small-fp16-256.mlpackage.zip
shasum -a 256 e5small-fp16-256.mlpackage.zip > SHA256SUMS
# stage the zip, SHA256SUMS, convert.py, requirements.txt and MODEL_CARD.md (as README.md) in one folder, then:
hf upload juhyeonni/tabieum-e5-small-coreml <folder> .
```

The app downloads a pinned commit, so a new upload never changes an installed app.

## License

Apache-2.0 for the code in this repository (see `LICENSE`). The converted weights are derived from `intfloat/multilingual-e5-small` and are released under the MIT License; see each model card.
