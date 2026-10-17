import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import make_card  # noqa: E402
from lib import SETS, dataset_sha256  # noqa: E402

REQUIRED = ["model", "version", "package", "tools", "device", "compute_units", "length", "dataset_sha256",
            "sets", "keyword_only", "comparison", "speed_ms", "memory_mb", "fidelity_cosine"]


def check_hits(label, hits, cutoffs):
    values = [hits[c] for c in cutoffs]
    assert values == sorted(values), f"{label}: recall must not shrink with the cutoff"
    assert 0 <= values[0] and values[-1] <= hits["n"], f"{label}: hits out of range"


def check_result(path):
    result = json.loads(path.read_text())
    missing = [key for key in REQUIRED if key not in result]
    assert not missing, f"missing keys {missing}"
    assert re.fullmatch(r"[0-9a-f]{64}", result["package"]["sha256"]), "package sha256"
    assert result["dataset_sha256"] == dataset_sha256(), "dataset changed since the results were recorded"
    for name in SETS:
        check_hits(f"sets.{name}", result["sets"][name], ["r1", "r3", "r10"])
        check_hits(f"keyword_only.{name}", result["keyword_only"][name], ["r1", "r3", "r10"])
        assert result["sets"][name]["n"] == result["keyword_only"][name]["n"], f"{name}: n differs"
        for option, sets in result["comparison"].items():
            assert sets[name]["n"] == result["sets"][name]["n"], f"{option}.{name}: n differs"
            check_hits(f"{option}.{name}", sets[name], ["r3"])
    assert all(ms > 0 for ms in result["speed_ms"].values()), "speed_ms"


def check_front_matter(path):
    match = re.match(r"---\n(.*?)\n---\n", path.read_text(), re.DOTALL)
    assert match, f"{path.name}: no front matter"
    for key in ("license:", "base_model:"):
        assert key in match.group(1), f"{path.name}: front matter lacks {key}"


def main():
    paths = sorted((ROOT / "eval" / "results").glob("*/*.json"))
    assert paths, "no results files"
    for path in paths:
        check_result(path)
        check_front_matter(ROOT / path.parent.name / "MODEL_CARD.md")
        print(f"ok {path.relative_to(ROOT)}")
    return make_card.main(check=True)


if __name__ == "__main__":
    sys.exit(main())
