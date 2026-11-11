"""
Human claim labels -> verifier accuracy and thresholds (report §14, §22.2).

The verifier labels each answer sentence ("claim") supported / partial /
unsupported from heuristics (similarity, word overlap, figures) and an
optional NLI model. Its thresholds (0.58 / 0.45) were set by inspection. This
tool checks them against people.

1. Export a sample of claims from a benchmark run for labelling:
       venv\\Scripts\\python -m evaluation.claim_labels export ^
           evaluation\\results\\system_benchmark_partial_XXXX.jsonl --n 120
   -> evaluation/results/claim_labels_<stamp>.csv
   Raters fill `human_label` with: supported | unsupported | unsure
   ("supported" = the evidence shown backs the sentence; judge the sentence
   against the evidence, not against your own knowledge). Leave `system_label`
   hidden or ignore it while labelling.

2. Measure the verifier and get suggested thresholds:
       venv\\Scripts\\python -m evaluation.claim_labels score evaluation\\results\\claim_labels_XXXX.csv
"""

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

from config import settings
from src.utils.io import load_jsonl, write_json
from src.verification.claims import is_fragment

RESULTS = settings.BASE_DIR / "evaluation" / "results"
FIELDS = ["claim_id", "question", "claim", "evidence", "human_label", "notes",
          "system_label", "support_score", "nli", "missing_numbers"]


def export(partial_files, n, seed):
    claims, seen = [], set()
    for path in partial_files:
        for row in load_jsonl(Path(path)):
            for i, c in enumerate(row.get("claims") or []):
                key = c["text"].strip().lower()
                # Older runs stored fragments (formula lines, "where ...") as claims.
                if key in seen or is_fragment(c["text"]):
                    continue
                seen.add(key)
                claims.append({
                    "claim_id": f"{row['id']}|{row['strategy']}|{i}",
                    "question": row.get("question", ""),
                    "claim": c["text"],
                    "evidence": c.get("best_evidence", ""),
                    "human_label": "", "notes": "",
                    "system_label": c.get("label"),
                    "support_score": c.get("support_score"),
                    "nli": c.get("nli") or "",
                    "missing_numbers": " ".join(c.get("missing_numbers") or []),
                })
    if not claims:
        raise SystemExit("No claims found. Use a partial .jsonl written after 9 Nov 2026 "
                         "(older runs did not store claims).")
    # Stratify so unsupported/partial claims are not drowned out by supported ones.
    rng = random.Random(seed)
    by_label = {}
    for c in claims:
        by_label.setdefault(c["system_label"], []).append(c)
    per = max(1, n // max(1, len(by_label)))
    sample = []
    for label, items in sorted(by_label.items()):
        sample += rng.sample(items, min(per, len(items)))
    rest = [c for c in claims if c not in sample]
    sample += rng.sample(rest, min(max(0, n - len(sample)), len(rest)))
    rng.shuffle(sample)

    out = RESULTS / f"claim_labels_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sample)
    print(f"{len(sample)} of {len(claims)} distinct claims -> {out}")
    print("Counts in sample:", {k: sum(c['system_label'] == k for c in sample) for k in by_label})


def _rates(pairs):
    """pairs: (predicted_unsupported, human_unsupported)."""
    tp = sum(p and h for p, h in pairs)
    fp = sum(p and not h for p, h in pairs)
    fn = sum(h and not p for p, h in pairs)
    tn = sum(not p and not h for p, h in pairs)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": round((tp + tn) / len(pairs), 3),
            "precision_unsupported": precision and round(precision, 3),
            "recall_unsupported": recall and round(recall, 3), "f1_unsupported": round(f1, 3)}


def score(path):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["human_label"].strip().lower() in ("supported", "unsupported")]
    if len(rows) < 20:
        raise SystemExit(f"Only {len(rows)} labelled claims; label at least 20 (ideally 100+).")
    human = [r["human_label"].strip().lower() == "unsupported" for r in rows]

    # As deployed: partial counts as supported for the unsupported claim rate.
    current = _rates([(r["system_label"] == "unsupported", h) for r, h in zip(rows, human)])

    # Sweep the support-score threshold (claims with missing figures or an NLI
    # contradiction are unsupported regardless, as in the verifier).
    sweep = []
    for t in [x / 100 for x in range(30, 81, 2)]:
        pred = [bool(r["missing_numbers"].strip()) or r["nli"] == "contradiction"
                or (r["nli"] != "entailment" and float(r["support_score"] or 0) < t) for r in rows]
        sweep.append({"threshold": t, **_rates(list(zip(pred, human)))})
    best = max(sweep, key=lambda s: (s["f1_unsupported"], s["accuracy"]))

    nli_rows = [(r["nli"], h) for r, h in zip(rows, human) if r["nli"]]
    nli = {
        "contradiction_precision": None, "entailment_said_supported_rate": None, "n": len(nli_rows),
    }
    contra = [h for v, h in nli_rows if v == "contradiction"]
    entail = [h for v, h in nli_rows if v == "entailment"]
    if contra:
        nli["contradiction_precision"] = round(sum(contra) / len(contra), 3)
    if entail:
        nli["entailment_said_supported_rate"] = round(sum(not h for h in entail) / len(entail), 3)

    report = {
        "labels_file": str(path), "labelled_claims": len(rows),
        "human_unsupported_share": round(sum(human) / len(human), 3),
        "verifier_as_deployed": current,
        "best_unsupported_threshold": best,
        "current_thresholds": {"partial": 0.45, "supported": 0.58},
        "nli": nli,
        "note": "Set PARTIAL_THRESHOLD in src/verification/claims.py to the suggested value only "
                "if it holds on a second labelled sample.",
    }
    out = Path(path).with_suffix(".score.json")
    write_json(out, report)
    print(json.dumps(report, indent=2))
    print("Saved:", out)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("partial_files", nargs="+")
    e.add_argument("--n", type=int, default=120)
    e.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score")
    s.add_argument("labels_csv")
    args = parser.parse_args()
    if args.cmd == "export":
        export(args.partial_files, args.n, args.seed)
    else:
        score(args.labels_csv)


if __name__ == "__main__":
    main()
