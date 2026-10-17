"""Retrieval quality of a Core ML embedding package on the bundled notes.

Usage: python eval/eval_coreml.py <package.mlpackage> [sequence_length] [CPU_AND_NE|CPU_ONLY|CPU_AND_GPU|ALL]
Prints R@1/3/10 for natural-language queries (hybrid: keyword ranking in fts.json fused with vectors by RRF, k=60).
"""
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).parent
package = sys.argv[1]
length = int(sys.argv[2]) if len(sys.argv) > 2 else 256
units = getattr(ct.ComputeUnit, sys.argv[3] if len(sys.argv) > 3 else "CPU_AND_NE")

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
model = ct.models.MLModel(package, compute_units=units)
keyword_rankings = json.loads((HERE / "fts.json").read_text())


def embed(text):
    batch = tokenizer([text], padding="max_length", max_length=length, truncation=True, return_tensors="np")
    feed = {"input_ids": batch["input_ids"].astype(np.int32), "attention_mask": batch["attention_mask"].astype(np.int32)}
    return model.predict(feed)["embedding"][0]


def normalize(rows):
    return rows / np.linalg.norm(rows, axis=1, keepdims=True).clip(1e-9)


def fuse(rankings, k=60):
    scores, order = {}, []
    for ranking in rankings:
        for offset, note in enumerate(ranking):
            if note not in scores:
                order.append(note)
            scores[note] = scores.get(note, 0) + 1 / (k + offset + 1)
    return [note for _, _, note in sorted((-scores[n], pos, n) for pos, n in enumerate(order))]


for name in ["en", "ko", "mixed"]:
    chunks = json.loads((HERE / f"chunks-{name}.json").read_text(encoding="utf-8"))
    data = json.loads((HERE / "data" / f"{name}.json").read_text(encoding="utf-8"))
    relevant = {q["id"]: (q["relevant"], q["type"]) for q in data["queries"]}
    docs = normalize(np.array([embed(d["text"]) for d in chunks["docs"]]))
    queries = normalize(np.array([embed(q["text"]) for q in chunks["queries"]]))
    similarity = queries @ docs.T
    owners = [d["note"] for d in chunks["docs"]]
    hits = {1: 0, 3: 0, 10: 0}
    total = 0
    for i, query in enumerate(chunks["queries"]):
        target, kind = relevant[query["id"]]
        if kind == "keyword":
            continue
        best = {}
        for j, score in enumerate(similarity[i]):
            best[owners[j]] = max(best.get(owners[j], -9), float(score))
        vectors = [n for n, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]
        fused = fuse([keyword_rankings[name][query["id"]][:50], vectors[:50]])[:50]
        rank = fused.index(target) + 1 if target in fused else 999
        total += 1
        for cutoff in hits:
            hits[cutoff] += rank <= cutoff
    print(f"{name:6s} " + "  ".join(f"R@{c} {hits[c] / total * 100:5.1f}" for c in hits) + f"  (n={total})")
