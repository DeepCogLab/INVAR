# INVAR: Stable to Form, Responsive to Substance

Code, benchmark data and key results for the paper
**"Stable to Form, Responsive to Substance: Evaluating Reliability in LLM-Based Simulation"** (under double-blind review).

- Project page: <https://anonymous.4open.science/w/invar-www27/>
- Anonymous code: <https://anonymous.4open.science/r/invar-www27/>

INVAR checks whether an LLM used as a simulated participant in a social-decision experiment
(1) keeps its decisions **stable** when only the presentation of a scenario changes, and
(2) **responds correctly** when a rule, permission, payoff or goal actually changes,
on the same scenarios.

| | |
|---|---|
| Source studies | 4 published social-decision studies with open materials |
| Task families | dictator allocation, ultimatum bargaining, trust return, sequential cooperation, public goods, third-party punishment |
| Condition pairs | 92 (14 kinds of change; 54 follow the original studies, 38 are extensions) |
| Quartets | 701 (two conditions x two equivalent presentations) |
| Unique inputs | 8,666 = 1,240 free decisions + 3,092 changed-fact + 2,888 unchanged-fact + 1,410 goal + 36 role questions |
| Language | Chinese prompts; English translation of a bilingual subset (32 pairs) |

## Repository layout

```
INVAR/
├── experiments/
│   ├── baseline_v2/INVAR_dataset/   # the benchmark: data, builders, evaluator, data card
│   ├── baseline_v2/                 # fact/goal checks, 8,666 inputs, generation + scoring (8 general models)
│   ├── social_cognition_v1/         # specialized models (behavior-prediction adapters, SocialR1, Socio) + bases
│   ├── shortpaper_20261111/         # Qwen3-14B / 30B-A3B main runs, goal discrimination, reasoning budgets
│   ├── confirmation_20261112/       # native reasoning off vs. on, all 92 pairs + 40 parametric pairs
│   ├── methods_v1/, methods_v2/     # helpers and the matched-exposure SFT / consistency (JS) trainer
│   ├── methods_v4/                  # held-out-source training: 2 models x 4 sources x 3 seeds
│   ├── mechanism_20261113_v2/       # checkpoint diagnostic of the consistency term (Figure 3)
│   └── robustness_20261115/         # bilingual / cross-family reasoning settings (Table 2, bottom)
├── results/
│   ├── paper_tables/                # accepted tables behind Tables 1-3 and Figures 2-3 (+ SHA provenance)
│   ├── supplement_20261111/         # goal discrimination, fact-goal pairing, native delivery, ...
│   ├── native_confirmation_20261112/# accepted behavior / goal / delivery tables and sampling statistics
│   ├── training_transfer_v4/        # per-source, per-seed training results
│   └── mechanism_20261113/          # per-checkpoint consistency-term diagnostic
├── paper/                           # make_figures.py regenerates Figures 2-3 and Tables 1-3
├── tools/build_writing_context.py   # builds results/paper_tables from the accepted run outputs
└── docs/                            # project page (GitHub Pages)
```

## The benchmark data

All benchmark files live in `experiments/baseline_v2/` (kept at their original paths so that every
frozen script still resolves them).

| File | Contents |
|---|---|
| `INVAR_dataset/data/prompts.jsonl` | 1,240 unique free-decision prompts; send only `messages` to the model |
| `INVAR_dataset/data/quartets.jsonl` | 701 quartets: prompt ids of A/B (original condition) and C/D (changed condition) |
| `INVAR_dataset/data/pairs.jsonl` | 92 condition pairs: source, intervention kind, adaptation notes |
| `INVAR_dataset/data/cases.jsonl` | 184 condition records: facts, action space and payoff table (for auditing; not sent to models) |
| `INVAR_dataset/controls/role_comprehension.jsonl` | 36 role-comprehension questions |
| `fact_quartets.jsonl`, `fact_checks.jsonl` | changed-fact and unchanged-fact questions with known answers |
| `goal_quartets.jsonl`, `goal_audit.jsonl` | goal questions; the correct answer is the full optimal action set |
| `inputs.jsonl` | all 8,666 unique model inputs with expected answers |
| `INVAR_dataset/DATA_CARD.md`, `INVAR_dataset/audit/` | data card, per-source rule audits, equivalence checks |
| `INVAR_dataset/provenance/sources.jsonl` | citations and links for the four source studies |

Third-party source materials (original papers, instructions, data files) are **not redistributed**;
`provenance/sources.jsonl` points to their public locations. Human choices are not used as correct
answers: free decisions are compared with each other, while fact and goal questions have known answers.

`experiments/robustness_20261115/frozen/` contains the bilingual subset (Chinese and English prompts),
and `experiments/confirmation_20261112/frozen/parameter/` the 40 parametric pairs whose two conditions
have disjoint optimal actions.

## Metrics

Let `p_X` be the distribution of normalized actions for prompt `X`, `mu_X` its mean and `W1` the
1-Wasserstein distance.

- **Presentation discrepancy** `E = 1/2 [W1(p_A, p_B) + W1(p_C, p_D)]` (lower = more stable)
- **Effect discrepancy** `G = |(mu_C - mu_A) - (mu_D - mu_B)|` (lower; `G <= 2E`)
- **Fact accuracy** on changed (Chg.) and unchanged (Unch.) facts, and **goal accuracy** over all attempts
  (invalid outputs count as errors; answers are never recovered from reasoning traces)

Scores are averaged within and then across pairs (pair-weighted); source- and task-weighted averages
are reported as sensitivity checks. Reference implementations: `INVAR_dataset/evaluate.py`
(`w1`, `normalize`, `parse_response`), `baseline_v2/score_v2.py`, and
`robustness_20261115/behavior_metrics.py`.

## Reproducing the paper's tables and figures (CPU only)

```bash
pip install numpy pandas matplotlib
cd paper
python make_figures.py
```

This reads only the accepted result tables in `results/` and rewrites `paper/tables/tab_*.tex`,
`paper/figures/fig2_scatter.pdf`, `paper/figures/fig3_training_bars.pdf` and `paper/numbers_used.txt`.
The regenerated tables are byte-identical to the ones in the submission. (Set `LIBERTINE_DIR` to a
Linux Libertine/Biolinum font folder to match the paper's typeface; otherwise a default font is used.)

| Paper item | Script | Data |
|---|---|---|
| Table 1, Figure 2 (28 configurations) | `paper/make_figures.py` | `results/paper_tables/all_model_configurations.csv`, `results/supplement_20261111/*goal_discrimination*.csv` |
| Table 2 top (reasoning off/on, 92 pairs) | same | `results/paper_tables/native_confirmation.csv`, `results/native_confirmation_20261112/accepted_delivery.csv` |
| Table 2 bottom (bilingual subset) | same | `experiments/robustness_20261115/results/summary_corrected.csv` |
| Table 3 (held-out-source training) | same | `results/paper_tables/training_summary.csv`, `results/training_transfer_v4/` |
| Figure 3 (checkpoint diagnostic) | same | `results/paper_tables/mechanism_checkpoint_means.csv`, `results/mechanism_20261113/` |

## Running models on INVAR (GPU)

The minimal pipeline needs only the standard library plus `vllm` for generation:

```bash
cd experiments/baseline_v2/INVAR_dataset
python make_run_plan.py --models my-model --tier all --repeats 32 --out runs/plan.jsonl
# send each prompt's `messages` in a fresh conversation, one call per plan row, and save raw outputs
python evaluate.py --help
python -m unittest discover -s tests       # CPU tests of the builder and evaluator
```

The full protocol used in the paper (structured JSON decoding restricted to legal actions,
temperature 0.7; 256 samples per free decision, 16 per goal question, 8 per fact question and
option order) is in `experiments/baseline_v2/run_structured.py` and `score_v2.py`; specialized
models use `experiments/social_cognition_v1/run_model.py`. Native-reasoning runs use
`experiments/confirmation_20261112/worker.py` and `score.py`; the bilingual settings use
`experiments/robustness_20261115/science_worker.py`, `score.py` and `rescore_terminal_v3_1.py`
(final v3 parser). Training uses `experiments/methods_v4/train_transfer.py` (LoRA rank 8; SFT on fact
and goal questions, with or without the consistency term defined in
`experiments/methods_v2/train_condition_anchor.py`), evaluated with `run_transfer_evaluation.py`,
`score_transfer.py` and `aggregate_transfer.py`.

Model paths in the scripts (e.g. `/workspace/models/...`) are placeholders for local copies of the
public Hugging Face checkpoints listed in the paper.

## Notes

- Every experiment folder is a snapshot of the code that was frozen and run; scripts are kept unchanged
  except that machine-specific paths were replaced by placeholders. Operational scripts (remote
  control, monitoring, synchronization) are omitted.
- Some protocol notes and summaries (`*.md`, `SUMMARY_zh.md`) are in Chinese, the language of the
  original prompts. The README files and this page describe everything needed to use the data.
- Raw model responses (millions of samples) and LoRA weights are not included because of their size.
- Limitations: all four sources were seen during development, so held-out results concern source
  exclusion rather than unseen sources; equivalent presentations were checked by programs and AI
  assistance, not yet by independent human reviewers; only open-weight models and single-decision
  calls are tested.

## License

Code: MIT. Benchmark data and results: CC BY 4.0. Rights in the original source studies remain with
their authors.
