"""
Blind answer grading -> per-strategy quality and rater agreement (report §22).

Each rater fills their own copy of the grading sheet (do not share scores
while grading), using evaluation/ANSWER_RUBRIC.md. Then:

    venv\\Scripts\\python -m evaluation.analyse_grading ^
        --key evaluation\\results\\grading_key_XXXX.json ^
        --benchmark evaluation\\results\\system_benchmark_XXXX.json ^
        rater1.csv rater2.csv

Output: mean 0-3 score per criterion and strategy (overall, in-domain,
out-of-domain), the share of answers a rater scored 0 for correctness, and
agreement between raters (exact, within 1 point, and quadratic-weighted
Cohen's kappa per criterion). Saved next to the first rater file.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

CRITERIA = ["correctness_0_3", "completeness_0_3", "relevance_0_3", "clarity_0_3"]


def load_sheet(path):
    rows = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        scores = {}
        for c in CRITERIA:
            value = (r.get(c) or "").strip()
            if value:
                if value not in {"0", "1", "2", "3"}:
                    raise SystemExit(f"{path}: {r['id']}/{r['system_code']} {c} = {value!r}; use 0-3")
                scores[c] = int(value)
        rows[(r["id"], r["system_code"])] = scores
    return rows


def weighted_kappa(a, b, k=4):
    """Quadratic-weighted Cohen's kappa for two raters on a 0..k-1 scale."""
    n = len(a)
    if n < 2:
        return None
    observed = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[x][y] += 1
    row = [sum(observed[i]) for i in range(k)]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * observed[i][j]
            den += w * row[i] * col[j] / n
    return round(1 - num / den, 3) if den else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sheets", nargs="+", help="one graded CSV per rater")
    parser.add_argument("--key", required=True, help="grading_key_*.json (system code -> strategy)")
    parser.add_argument("--benchmark", help="system_benchmark_*.json, to split in/out of domain")
    args = parser.parse_args()

    key = json.load(open(args.key, encoding="utf-8"))
    code_to_strategy = {code: strategy for strategy, code in key.items()}
    sets = {}
    if args.benchmark:
        for r in json.load(open(args.benchmark, encoding="utf-8"))["rows"]:
            sets[r["id"]] = r["set"]

    raters = [load_sheet(p) for p in args.sheets]
    graded = [{k: v for k, v in r.items() if len(v) == len(CRITERIA)} for r in raters]
    for path, g, r in zip(args.sheets, graded, raters):
        print(f"{Path(path).name}: {len(g)} of {len(r)} answers fully graded")

    # Mean over raters per answer, then per strategy.
    items = set().union(*(g.keys() for g in graded))
    per_answer = {}
    for item in items:
        scored = [g[item] for g in graded if item in g]
        per_answer[item] = {c: statistics.mean(s[c] for s in scored) for c in CRITERIA}

    report = {"raters": len(args.sheets), "answers_graded": len(per_answer), "by_strategy": {}}
    for code, strategy in sorted(code_to_strategy.items(), key=lambda x: x[1]):
        entry = {}
        for subset in ("all", "in_domain", "out_of_domain"):
            mine = [v for (qid, c), v in per_answer.items() if c == code
                    and (subset == "all" or sets.get(qid) == subset)]
            if mine:
                entry[subset] = {"n": len(mine), **{c.replace("_0_3", ""): round(statistics.mean(
                    m[c] for m in mine), 2) for c in CRITERIA},
                    "overall": round(statistics.mean(statistics.mean(m.values()) for m in mine), 2),
                    "wrong_answers_share": round(sum(m["correctness_0_3"] < 1 for m in mine) / len(mine), 3)}
        report["by_strategy"][strategy] = entry

    if len(graded) >= 2:
        both = sorted(set(graded[0]) & set(graded[1]))
        agreement = {"answers_rated_by_both": len(both)}
        for c in CRITERIA:
            a = [graded[0][i][c] for i in both]
            b = [graded[1][i][c] for i in both]
            if a:
                agreement[c.replace("_0_3", "")] = {
                    "exact": round(sum(x == y for x, y in zip(a, b)) / len(a), 3),
                    "within_1": round(sum(abs(x - y) <= 1 for x, y in zip(a, b)) / len(a), 3),
                    "weighted_kappa": weighted_kappa(a, b),
                }
        report["rater_agreement"] = agreement

    print(f"\n{'strategy':<15}{'n':>4}{'correct':>9}{'complete':>10}{'relevant':>10}{'clear':>7}"
          f"{'overall':>9}{'wrong':>7}")
    for strategy, entry in report["by_strategy"].items():
        a = entry.get("all")
        if a:
            print(f"{strategy:<15}{a['n']:>4}{a['correctness']:>9}{a['completeness']:>10}"
                  f"{a['relevance']:>10}{a['clarity']:>7}{a['overall']:>9}{a['wrong_answers_share']:>7}")
    if "rater_agreement" in report:
        print("\nRater agreement (weighted kappa: >0.6 good, 0.4-0.6 moderate, <0.4 revisit the rubric):")
        for c, v in report["rater_agreement"].items():
            if isinstance(v, dict):
                print(f"  {c:<13} exact {v['exact']}  within 1 {v['within_1']}  kappa {v['weighted_kappa']}")

    out = Path(args.sheets[0]).with_name(Path(args.sheets[0]).stem + "_analysis.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
