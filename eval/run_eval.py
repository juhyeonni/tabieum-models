import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import coremltools as ct
import numpy as np
from lib import SETS, count_hits, dataset_sha256, fuse, keyword_only_hits, keyword_rankings, load_set
from transformers import AutoTokenizer


def embedder(package, length, units):
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    model = ct.models.MLModel(package, compute_units=getattr(ct.ComputeUnit, units))

    def embed(text):
        batch = tokenizer([text], padding="max_length", max_length=length, truncation=True, return_tensors="np")
        feed = {"input_ids": batch["input_ids"].astype(np.int32), "attention_mask": batch["attention_mask"].astype(np.int32)}
        return model.predict(feed)["embedding"][0]

    return embed


def normalize(rows):
    return rows / np.linalg.norm(rows, axis=1, keepdims=True).clip(1e-9)


def evaluate_set(name, embed):
    chunks, relevant = load_set(name)
    docs = normalize(np.array([embed(d["text"]) for d in chunks["docs"]]))
    queries = normalize(np.array([embed(q["text"]) for q in chunks["queries"]]))
    similarity = queries @ docs.T
    owners = [d["note"] for d in chunks["docs"]]
    rankings = keyword_rankings()[name]
    ranks = []
    for i, query in enumerate(chunks["queries"]):
        target, kind = relevant[query["id"]]
        if kind == "keyword":
            continue
        best = {}
        for j, score in enumerate(similarity[i]):
            best[owners[j]] = max(best.get(owners[j], -9), float(score))
        vectors = [n for n, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]
        fused = fuse([rankings[query["id"]][:50], vectors[:50]])[:50]
        ranks.append(fused.index(target) + 1 if target in fused else 999)
    return count_hits(ranks)


def chip():
    return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()


def zip_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measured(args):
    embed = embedder(args.package, args.length, args.units)
    return {
        "tools": {name: version(name) for name in ("torch", "coremltools", "transformers")},
        "device": {"chip": chip(), "macos": platform.mac_ver()[0]},
        "compute_units": args.units,
        "length": args.length,
        "dataset_sha256": dataset_sha256(),
        "sets": {name: evaluate_set(name, embed) for name in SETS},
        "keyword_only": {name: keyword_only_hits(name) for name in SETS},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--units", default="CPU_AND_NE")
    parser.add_argument("--zip")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = measured(args)
    for name, hits in result["sets"].items():
        print(f"{name:6s} " + "  ".join(f"R@{c} {hits[f'r{c}'] / hits['n'] * 100:5.1f}" for c in (1, 3, 10)) + f"  (n={hits['n']})")
    if not args.out:
        return
    out = Path(args.out)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(result)
    if args.zip:
        existing["package"] = {"file": Path(args.zip).name, "sha256": zip_sha256(args.zip)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
