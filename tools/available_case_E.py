"""Pair-weighted E over quartets with a defined E (available-case), bilingual subset.

The frozen analysis leaves the panel E undefined when any quartet has a prompt with no
valid free decision (SocialR1-8B, Chinese, reasoning off: 2 of 64 quartets, one pair).
This script recomputes pair-weighted E from the accepted per-quartet scores using only
quartets whose E is defined, with the same aggregation as analyze.py (mean within a pair,
then mean over pairs). For every cell whose frozen E is defined, the result equals the
frozen value, which the script asserts.

Usage: python tools/available_case_E.py <corrected_v3/runs dir> <summary_corrected.csv> <out.csv>
"""
import collections
import csv
import statistics
import sys
from pathlib import Path

runs, summary, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
frozen = {(r["model"], r["language"], r["arm"]): r for r in csv.DictReader(open(summary, encoding="utf-8-sig"))}
rows = []
for model in ["qwen3_8b", "socialr1_8b"]:
    q = list(csv.DictReader(open(runs / model / "behavior_quartet_scores.csv", encoding="utf-8-sig")))
    for (lang, arm) in sorted({(r["language"], r["arm"]) for r in q}):
        sel = [r for r in q if r["language"] == lang and r["arm"] == arm]
        by = collections.defaultdict(list)
        for r in sel:
            if r["E_conditional"]:
                by[r["pair_id"]].append(float(r["E_conditional"]))
        e = statistics.mean(statistics.mean(v) for v in by.values())
        f = frozen[(model, lang, arm)]["E_conditional"]
        if f:
            assert abs(float(f) - e) < 1e-9, (model, lang, arm, f, e)
        rows.append(dict(model=model, language=lang, arm=arm, E_available_case=e,
                         quartets_used=sum(len(v) for v in by.values()), quartets_total=len(sel),
                         pairs_used=len(by), pairs_total=len({r["pair_id"] for r in sel}),
                         frozen_E_defined=bool(f)))
with open(out, "w", newline="", encoding="utf-8") as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
