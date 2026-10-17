import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
NAME = "intfloat/multilingual-e5-small"
LENGTH = 256
SENTENCES = [
    "query: Which morning is the polishing session for my molars?",
    "passage: Dr. Okafor, cleaning, Thursday March 14 at 9:30. Bring insurance card.",
    "query: 치과 예약이 언제였지?",
    "passage: 오늘 회의에서 릴리스 일정을 논의했다. 서명과 공증은 다음 주까지 끝내기로 했다.",
    "passage: 今日は会議でリリースの日程について話し合いました。",
    "passage: Notes on semantic search: combine FTS5 ranking with vectors, 검색 품질은 한국어에서 약하다.",
]
UNITS = ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]
INT32, FLOAT16 = 131104, 65552


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_zip(path):
    digest = sha256(path)
    recorded = {json.loads(p.read_text())["package"]["sha256"] for p in (ROOT / "eval" / "results").glob("*/*.json")}
    assert digest in recorded, f"{path.name}: sha256 {digest} is not in any results file"
    sums = path.parent / "SHA256SUMS"
    if sums.exists():
        assert f"{digest}  {path.name}" in sums.read_text(), f"{path.name}: sha256 differs from {sums}"
    print(f"ok zip sha256 {digest[:12]}")


def unpack(path, target):
    with zipfile.ZipFile(path) as archive:
        archive.extractall(target)
    return next(Path(target).glob("**/*.mlpackage"))


def check_contract(package):
    description = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY).get_spec().description
    inputs = {i.name: i.type.multiArrayType for i in description.input}
    outputs = {o.name: o.type.multiArrayType for o in description.output}
    assert set(inputs) == {"input_ids", "attention_mask"}, f"inputs {sorted(inputs)}"
    for name, array in inputs.items():
        assert list(array.shape) == [1, LENGTH] and array.dataType == INT32, f"{name}: {list(array.shape)} {array.dataType}"
    assert set(outputs) == {"embedding"}, f"outputs {sorted(outputs)}"
    assert list(outputs["embedding"].shape) == [1, 384] and outputs["embedding"].dataType == FLOAT16, "embedding"
    print("ok contract")


def reference_vectors(tokenizer, batches):
    model = AutoModel.from_pretrained(NAME).eval()
    vectors = []
    for batch in batches:
        with torch.no_grad():
            hidden = model(**{k: torch.from_numpy(v).long() for k, v in batch.items()}).last_hidden_state
        mask = torch.from_numpy(batch["attention_mask"]).unsqueeze(-1).float()
        vectors.append(torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)[0].numpy())
    return vectors


def predict(model, batch):
    feed = {"input_ids": batch["input_ids"].astype(np.int32), "attention_mask": batch["attention_mask"].astype(np.int32)}
    return model.predict(feed)["embedding"][0].astype(np.float32)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def check_vectors(package):
    tokenizer = AutoTokenizer.from_pretrained(NAME)
    batches = [tokenizer([s], padding="max_length", max_length=LENGTH, truncation=True, return_tensors="np") for s in SENTENCES]
    expected = reference_vectors(tokenizer, batches)
    for units in UNITS:
        model = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, units))
        vectors = [predict(model, batch) for batch in batches]
        assert not any(np.isnan(v).any() for v in vectors), f"{units}: NaN in output"
        if units != "CPU_ONLY":
            continue
        worst = min(cosine(v, e) for v, e in zip(vectors, expected))
        assert worst >= 0.999, f"{units}: cosine {worst:.5f} below 0.999"
        print(f"ok fidelity {units} worst cosine {worst:.5f}")
    print("ok no NaN on " + ", ".join(UNITS))


def main():
    if len(sys.argv) < 2:
        print("stage 2 skipped: pass a .mlpackage or its .zip to check_model.py")
        return 0
    source = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as scratch:
        if source.suffix == ".zip":
            check_zip(source)
            package = unpack(source, scratch)
        else:
            package = source
        check_contract(package)
        check_vectors(package)
    return 0


if __name__ == "__main__":
    sys.exit(main())
