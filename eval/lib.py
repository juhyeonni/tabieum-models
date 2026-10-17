import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
SETS = ["en", "ko", "mixed"]
CUTOFFS = (1, 3, 10)


def dataset_files():
    return sorted(HERE.glob("data/*.json")) + sorted(HERE.glob("chunks-*.json")) + [HERE / "fts.json"]


def dataset_sha256():
    digest = hashlib.sha256()
    for path in dataset_files():
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_set(name):
    chunks = json.loads((HERE / f"chunks-{name}.json").read_text(encoding="utf-8"))
    data = json.loads((HERE / "data" / f"{name}.json").read_text(encoding="utf-8"))
    relevant = {q["id"]: (q["relevant"], q["type"]) for q in data["queries"]}
    return chunks, relevant


def keyword_rankings():
    return json.loads((HERE / "fts.json").read_text())


def fuse(rankings, k=60):
    scores, order = {}, []
    for ranking in rankings:
        for offset, note in enumerate(ranking):
            if note not in scores:
                order.append(note)
            scores[note] = scores.get(note, 0) + 1 / (k + offset + 1)
    return [note for _, _, note in sorted((-scores[n], pos, n) for pos, n in enumerate(order))]


def count_hits(ranks):
    return {"n": len(ranks), **{f"r{c}": sum(rank <= c for rank in ranks) for c in CUTOFFS}}


def keyword_only_hits(name):
    chunks, relevant = load_set(name)
    rankings = keyword_rankings()[name]
    ranks = []
    for query in chunks["queries"]:
        target, kind = relevant[query["id"]]
        if kind == "keyword":
            continue
        ranking = rankings[query["id"]][:50]
        ranks.append(ranking.index(target) + 1 if target in ranking else 999)
    return count_hits(ranks)


def wilson(hits, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denom = 1 + z * z / n
    return ((centre - spread) / denom, (centre + spread) / denom)
