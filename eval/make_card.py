import json
import re
import sys

from lib import CUTOFFS, HERE, SETS, wilson

ROOT = HERE.parent
LABELS = {"en": "English", "ko": "Korean", "mixed": "Mixed Korean/English"}
UNITS = {"cpu_only": "`cpuOnly`", "cpu_and_ne": "`cpuAndNeuralEngine`"}


def pct(hits, n):
    return f"{hits / n * 100:.1f}"


def with_interval(hits, n):
    low, high = wilson(hits, n)
    return f"{pct(hits, n)} ({low * 100:.0f}–{high * 100:.0f})"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def speed_block(r):
    device = r["device"]
    fidelity = r["fidelity_cosine"]
    memory = r["memory_mb"]
    rows = [[UNITS[unit], f"{ms:.1f}"] for unit, ms in r["speed_ms"].items()]
    return "\n".join([
        f"{device['chip']}, macOS {device['macos']}, one text per call, Core ML via Swift. These are our own measurements, not a standard benchmark.",
        "",
        table(["Compute units", f"ms per text (length {r['length']})"], rows),
        "",
        f"- **Memory** (`footprint` of the process, ANE included): about {memory['app'] + memory['ane']} MB "
        f"({memory['app']} MB in the app, {memory['ane']} MB owned by the Neural Engine). "
        "The first load compiles for the Neural Engine and takes a few seconds.",
        f"- **Fidelity:** cosine similarity to the PyTorch model on the same text is {fidelity['cpu_and_ne']} on the Neural Engine "
        f"and {fidelity['cpu_only']:.5f} on CPU.",
    ])


def retrieval_block(r):
    rows = [[LABELS[s], str(r["sets"][s]["n"])] + [with_interval(r["sets"][s][f"r{c}"], r["sets"][s]["n"]) for c in CUTOFFS] for s in SETS]
    own = table(["Set", "Queries"] + [f"R@{c} % (95% CI)" for c in CUTOFFS], rows)
    versus = [["This model (hybrid search)"] + [pct(r["sets"][s]["r3"], r["sets"][s]["n"]) for s in SETS]]
    versus.append(["Keyword search alone (tabieum FTS5)"] + [pct(r["keyword_only"][s]["r3"], r["keyword_only"][s]["n"]) for s in SETS])
    for name, sets in r["comparison"].items():
        versus.append([name] + [pct(sets[s]["r3"], sets[s]["n"]) for s in SETS])
    return "\n".join([
        "Hybrid search (SQLite FTS5 keyword ranking fused with these vectors by reciprocal rank fusion), natural-language queries. "
        "The notes and queries are synthetic and AI-written, so compare models against each other rather than reading absolute numbers.",
        "",
        own,
        "",
        "Recall at 3 (%) against other options on the same queries. Keyword search requires every query word to appear, so it finds nothing for these paraphrased queries.",
        "",
        table(["Option"] + [LABELS[s] for s in SETS], versus),
    ])


BLOCKS = {"speed": speed_block, "retrieval": retrieval_block}


def render(card, results):
    for name, build in BLOCKS.items():
        pattern = re.compile(rf"(<!-- results:begin {name} -->\n).*?(<!-- results:end {name} -->)", re.DOTALL)
        if not pattern.search(card):
            raise SystemExit(f"missing results:{name} markers")
        card = pattern.sub(lambda m: m.group(1) + build(results) + "\n" + m.group(2), card)
    return card


def main(check=False):
    status = 0
    for path in sorted(HERE.glob("results/*/*.json")):
        card_path = ROOT / path.parent.name / "MODEL_CARD.md"
        card = card_path.read_text()
        updated = render(card, json.loads(path.read_text()))
        if updated == card:
            continue
        if check:
            print(f"{card_path.relative_to(ROOT)} is out of date with {path.relative_to(ROOT)}")
            status = 1
            continue
        card_path.write_text(updated)
        print(f"updated {card_path.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
