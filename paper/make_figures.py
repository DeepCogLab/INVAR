# -*- coding: utf-8 -*-
"""Build every data figure and table fragment of the INVAR paper.

All numbers are read from accepted result tables of the project; nothing is typed
by hand.  Run:  python make_figures.py   (then latexmk -pdf main.tex)

Outputs
  figures/fig2_scatter.pdf        RQ1: goal vs stability and goal vs changed facts, all 28
  figures/fig3_training_bars.pdf  RQ2: reduction in discrepancy from the consistency term
  tables/tab_main.tex             all 28 configurations (full width)
  tables/tab_reasoning.tex        native reasoning off/on
  tables/tab_training.tex         matched-exposure fine-tuning
  numbers_used.txt                plain-text list of the plotted numbers
"""
import glob
import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
P = lambda *a: os.path.join(ROOT, *a)

SRC = {
    "models": P("results", "paper_tables", "all_model_configurations.csv"),
    "native": P("results", "paper_tables", "native_confirmation.csv"),
    "training": P("results", "paper_tables", "training_summary.csv"),
    "delivery": P("results", "native_confirmation_20261112", "accepted_delivery.csv"),
    "discr_old": P("results", "supplement_20261111", "goal_discrimination_summary.csv"),
    "discr_new": P("results", "supplement_20261111", "new_goal_discrimination.csv"),
    "robust": P("experiments", "robustness_20261115", "results", "summary_corrected.csv"),
    "seeds": P("results", "training_transfer_v4", "all_sources_seeds_JS_minus_SFT.csv"),
}

# ------------------------------------------------------------------ style
for _d in [os.environ.get("LIBERTINE_DIR", "/usr/share/texlive/texmf-dist/fonts/opentype/public/libertine")]:
    for _f in glob.glob(os.path.join(_d, "LinLibertine_*.otf")) + \
            glob.glob(os.path.join(_d, "LinBiolinum_*.otf")):
        try:
            font_manager.fontManager.addfont(_f)
        except Exception:
            pass

COL_W = 3.33
INK = "#1E2A38"
NAVY = "#2F4B7C"
STEEL = "#A9BCD0"
TEAL = "#1F7A72"
CORAL = "#C0473A"
GRID = "#E3E7EC"

plt.rcParams.update({
    "font.family": "Linux Biolinum O",
    "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.4,
    "xtick.labelsize": 6.6, "ytick.labelsize": 7.0, "legend.fontsize": 6.8,
    "axes.linewidth": 0.5, "axes.edgecolor": "#55606E",
    "xtick.major.width": 0.5, "ytick.major.width": 0.0,
    "xtick.major.size": 2.0, "ytick.major.size": 0.0,
    "xtick.color": "#55606E", "ytick.color": INK,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.pad_inches": 0.015, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def _csv(p):
    return pd.read_csv(p, encoding="utf-8-sig")


def _save(fig, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=300)
    plt.close(fig)


def _write(path, s):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


NUMBERS = []


def note(*a):
    NUMBERS.append(" ".join(str(x) for x in a))


NAMES = {
    "deepseek_llm7b_chat": "DeepSeek-LLM-7B", "glm4_9b": "GLM-4-9B",
    "llama31_8b_instruct": "Llama-3.1-8B", "mistral7b_instruct_v03": "Mistral-7B-v0.3",
    "qwen25_14b_instruct": "Qwen2.5-14B", "qwen25_7b_base": "Qwen2.5-7B-Base",
    "qwen25_7b_instruct": "Qwen2.5-7B", "qwen3_8b": "Qwen3-8B",
    "qwen3_14b": "Qwen3-14B", "qwen3_30b_a3b_gptq": "Qwen3-30B-A3B",
    "smoltaur": "Smoltaur-0.4B", "qwentaur06": "Qwentaur-0.6B", "olmotaur": "Olmotaur-1B",
    "qwentaur17": "Qwentaur-1.7B", "qwentaur4": "Qwentaur-4B", "qwentaur8": "Qwentaur-8B",
    "socialr1_4": "SocialR1-4B", "socialr1_8": "SocialR1-8B", "socio": "Socio-8B",
    "smoltaur_base": "SmolLM2-0.4B", "qwentaur06_base": "Qwen3-0.6B-Base",
    "olmotaur_base": "OLMo-2-1B", "qwentaur17_base": "Qwen3-1.7B-Base",
    "qwentaur4_base": "Qwen3-4B-Base", "qwentaur8_base": "Qwen3-8B-Base",
    "qwen3_4b_control": "Qwen3-4B", "qwen3_8b_control": "Qwen3-8B (rerun)",
    "sotopia_base": "Mistral-7B-v0.1",
}
SPECIALIZED = {"smoltaur", "qwentaur06", "olmotaur", "qwentaur17", "qwentaur4", "qwentaur8",
               "socialr1_4", "socialr1_8", "socio"}
BASES = {"smoltaur_base", "qwentaur06_base", "olmotaur_base", "qwentaur17_base",
         "qwentaur4_base", "qwentaur8_base", "qwen3_4b_control", "qwen3_8b_control"}


def load_models():
    d = _csv(SRC["models"]).set_index("model")
    d["name"] = [NAMES[m] for m in d.index]
    return d


def load_discrimination():
    a = _csv(SRC["discr_old"])
    a = a[(a.source == "all") & (a.campaign.isin(["baseline_v2", "social_cognition_v1"]))]
    out = dict(zip(a.model, a.both_correct_independent))
    b = _csv(SRC["discr_new"])
    b = b[(b.arm == "original") & (b.group == "all")]
    out.update(dict(zip(b.job, b.both_correct_independent)))
    return out


# ------------------------------------------------------------------ number formats
def f3(x):
    return ("%.3f" % x).lstrip("0") if x < 1 else "%.3f" % x


def pc(x, nd=1):
    return ("%." + str(nd) + "f") % (100 * x)


GOODC, BADC = "goodc", "badc"


def shade(v, lo, hi, good_low=True, cap=40):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    x = (v - lo) / (hi - lo)
    x = min(max(x, 0.0), 1.0)
    if good_low:
        x = 1 - x
    if x >= 0.5:
        return r"\cellcolor{goodc!%d}" % int(round((x - 0.5) * 2 * cap))
    return r"\cellcolor{badc!%d}" % int(round((0.5 - x) * 2 * cap))


# ================================================================== Table 1
def table_main(out):
    d = load_models()
    left = [
        ("General-purpose, instruction-tuned", [
            "deepseek_llm7b_chat", "glm4_9b", "llama31_8b_instruct", "mistral7b_instruct_v03",
            "qwen25_7b_instruct", "qwen25_14b_instruct", "qwen3_8b", "qwen3_14b",
            "qwen3_30b_a3b_gptq"]),
        ("Social-reasoning models (control in grey)", [
            "qwen3_4b_control", "socialr1_4", "qwen3_8b_control", "socialr1_8", "socio"]),
    ]
    right = [
        ("Behavior-prediction adapters (base in grey)", [
            "smoltaur_base", "smoltaur", "qwentaur06_base", "qwentaur06", "olmotaur_base",
            "olmotaur", "qwentaur17_base", "qwentaur17", "qwentaur4_base", "qwentaur4",
            "qwentaur8_base", "qwentaur8"]),
        ("Additional pretrained bases", ["qwen25_7b_base", "sotopia_base"]),
    ]
    cols = ["E", "G", "changed_fact_U", "stable_fact_U", "goal_Q"]
    best = {"E": d.E.min(), "G": d.G.min(), "changed_fact_U": d.changed_fact_U.max(),
            "stable_fact_U": d.stable_fact_U.max(), "goal_Q": d.goal_Q.max()}

    def cell(r, c):
        v = r[c]
        if c in ("E", "G"):
            s = shade(math.log(v), math.log(0.011), math.log(0.2)) + f3(v)
        elif c == "goal_Q":
            s = shade(v, 0.45, 1.0, False) + pc(v)
        else:
            s = shade(v, 0.5, 1.0, False) + pc(v)
        if abs(v - best[c]) < 1e-12:
            s = s.replace(f3(v) if c in ("E", "G") else pc(v),
                          r"\textbf{%s}" % (f3(v) if c in ("E", "G") else pc(v)))
        return s

    def rows(blocks):
        L = []
        for title, ms in blocks:
            L.append(r"\multicolumn{6}{@{}l}{\textit{%s}}" % title)
            for m in ms:
                r = d.loc[m]
                nm = r["name"].replace(" (rerun)", "")
                if m in BASES:
                    nm = r"\textcolor{black!50}{%s}" % nm
                elif m in SPECIALIZED:
                    nm = r"\,$\hookrightarrow$\,%s" % nm
                L.append(nm + " & " + " & ".join(cell(r, c) for c in cols))
                note("T1", m, *["%.4f" % r[c] for c in cols])
        return L

    L, R = rows(left), rows(right)
    n = max(len(L), len(R))
    L += [" & & & & & "] * (n - len(L))
    R += [" & & & & & "] * (n - len(R))
    body = []
    for a, b in zip(L, R):
        a2 = a if a.startswith(r"\multicolumn") else a
        b2 = b
        if a.startswith(r"\multicolumn") and b.startswith(r"\multicolumn"):
            body.append(a2 + " & & " + b2 + r" \\")
        elif a.startswith(r"\multicolumn"):
            body.append(a2 + " & & " + b2 + r" \\")
        elif b.startswith(r"\multicolumn"):
            body.append(a2 + " & & " + b2 + r" \\")
        else:
            body.append(a2 + " & & " + b2 + r" \\")
    head = (r"Model & $E\!\downarrow$ & $G\!\downarrow$ & Chg.$\uparrow$ & Unch.$\uparrow$"
            r" & Goal$\uparrow$")
    grp = (r" & \multicolumn{2}{c}{Stable to form} & \multicolumn{3}{c}{Responsive to substance}")
    s = (r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{\hspace{6pt}}c@{\hspace{6pt}}lccccc@{}}" "\n"
         r"\toprule" "\n"
         + grp + " & & " + grp + r" \\" "\n"
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){9-10}\cmidrule(l){11-13}" "\n"
         + head + " & & " + head + r" \\" "\n" r"\midrule" "\n"
         + "\n".join(body) + "\n" r"\bottomrule" "\n" r"\end{tabular*}" "\n")
    _write(out, s)


# ================================================================== Table 2
def table_reasoning(out):
    nat = _csv(SRC["native"])
    dl = _csv(SRC["delivery"])
    rb = _csv(SRC["robust"])

    def pair(off, on, kind, bounds=None):
        """kind: 'acc' (higher better) or 'E' (lower better). Better value bold;
        an E that rises with reasoning is marked in badc.  If the pre-registered
        conditional E is undefined (some prompt has no valid sample), print the
        missing-mass bounds instead of a bare dash."""
        if kind == "acc":
            fo, fn = pc(off), pc(on, 2 if on > 0.995 else 1)
            if on >= off:
                return fo, r"\textbf{%s}" % fn
            return r"\textbf{%s}" % fo, r"\textcolor{badc}{%s}" % fn
        if off is None or np.isnan(off):
            if bounds is not None:
                return r"{\scriptsize %s--%s}$^\dagger$" % (f3(bounds[0]), f3(bounds[1])), f3(on)
            return "--", f3(on)
        fo, fn = f3(off), f3(on)
        if on <= off:
            return fo, r"\textbf{%s}" % fn
        return r"\textbf{%s}" % fo, r"\textcolor{badc}{%s}" % fn

    def valid(job, arm):
        r = dl[(dl.job == job) & (dl.stage == "canonical") & (dl.arm == arm) &
               (dl.task_type == "behavior")].iloc[0]
        return r.valid / r.n

    L = []
    names = {"qwen3_14b": "Qwen3-14B", "qwen3_30b_a3b_gptq": "Qwen3-30B-A3B"}
    for job in ["qwen3_14b", "qwen3_30b_a3b_gptq"]:
        v0, v1 = valid(job, "native_no_thinking"), valid(job, "native_thinking")
        note("T2a", job, "valid behavior off=%.5f on=%.5f" % (v0, v1))
        for k, w in enumerate(["pair_equal", "source_equal", "task_equal"]):
            a = nat[(nat.job == job) & (nat.weighting == w)]
            off = a[a.arm == "native_no_thinking"].iloc[0]
            on = a[a.arm == "native_thinking"].iloc[0]
            g0, g1 = pair(off.Q_all_attempts, on.Q_all_attempts, "acc")
            e0, e1 = pair(off.E_conditional, on.E_conditional, "E")
            note("T2a", job, w, "Q %.5f %.5f E %.5f %.5f" % (
                off.Q_all_attempts, on.Q_all_attempts, off.E_conditional, on.E_conditional))
            first = (r"\multirow{3}{*}{%s}" % names[job]) if k == 0 else ""
            vv = (r"\multirow{3}{*}{%s} & \multirow{3}{*}{%s}" % (pc(v0), pc(v1))) \
                if k == 0 else " & "
            L.append("%s & %s & %s & %s & %s & %s & %s \\\\" % (
                first, w.split("_")[0], g0, g1, e0, e1, vv))
        L.append(r"\midrule")
    L.append(r"\multicolumn{9}{@{}l}{\textit{Bilingual subset (32 pairs; 13 with goal questions)}} \\")
    names2 = {"qwen3_8b": "Qwen3-8B", "socialr1_8b": "SocialR1-8B"}
    for model in ["qwen3_8b", "socialr1_8b"]:
        for k, lang in enumerate(["zh", "en"]):
            a = rb[(rb.model == model) & (rb.language == lang)]
            off = a[a.arm == "native_off"].iloc[0]
            on = a[a.arm == "native_on"].iloc[0]
            g0, g1 = pair(off.Q, on.Q, "acc")
            e0, e1 = pair(off.E_conditional, on.E_conditional, "E",
                          bounds=(off.E_missing_low, off.E_missing_high))
            b0, b1 = pc(off.behavior_delivery), pc(on.behavior_delivery)
            if on.behavior_delivery < off.behavior_delivery - 0.02:
                b1 = r"\textcolor{badc}{%s}" % b1
            note("T2b", model, lang, "Q %.5f %.5f E %s %.5f valid %.5f %.5f" % (
                off.Q, on.Q, off.E_conditional, on.E_conditional,
                off.behavior_delivery, on.behavior_delivery))
            first = (r"\multirow{2}{*}{%s}" % names2[model]) if k == 0 else ""
            L.append("%s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
                first, {"zh": "Chinese", "en": "English"}[lang], g0, g1, e0, e1, b0, b1))
    s = (r"\begin{tabular}{@{}llcccccc@{}}" "\n" r"\toprule" "\n"
         r" & & \multicolumn{2}{c}{Goal acc. (\%)$\uparrow$} & \multicolumn{2}{c}{$E\!\downarrow$}"
         r" & \multicolumn{2}{c}{Valid (\%)} \\" "\n"
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}" "\n"
         r"Model & Setting & off & on & off & on & off & on \\" "\n" r"\midrule" "\n"
         + "\n".join(L) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    s = s.replace(r"\multicolumn{9}", r"\multicolumn{8}")
    _write(out, s)


# ================================================================== Table 3
def table_training(out):
    t = _csv(SRC["training"])
    meth = [("baseline", "Original"), ("anchor_sft", "SFT"), ("anchor_js", "+ consistency")]
    cols = [("pair_weighted", "behavior:E", "E"), ("pair_weighted", "behavior:G", "E"),
            ("pair_weighted", "changed_fact:accuracy", "acc"),
            ("pair_weighted", "stable_fact:accuracy", "acc"),
            ("pair_weighted", "goal:accuracy", "acc"),
            ("source_macro", "behavior:E", "E")]
    L = []
    for model, nm in [("qwen3_8b", "Qwen3-8B"), ("qwen25_14b_instruct", "Qwen2.5-14B")]:
        vals = {}
        for agg, c, kind in cols:
            for m, _ in meth:
                vals[(agg, c, m)] = t[(t.model == model) & (t.aggregation == agg) &
                                      (t.method == m)][c].iloc[0]
        for i, (m, lab) in enumerate(meth):
            cells = []
            for agg, c, kind in cols:
                v = vals[(agg, c, m)]
                group = [vals[(agg, c, mm)] for mm, _ in meth]
                isbest = (v == min(group)) if kind == "E" else (v == max(group))
                txt = f3(v) if kind == "E" else pc(v)
                cells.append(r"\textbf{%s}" % txt if isbest else txt)
            note("T3", model, m, *["%.5f" % vals[(a, c, m)] for a, c, _ in cols])
            first = (r"\multirow{3}{*}{%s}" % nm) if i == 0 else ""
            L.append("%s & %s & %s \\\\" % (first, lab, " & ".join(cells)))
        L.append(r"\midrule")
    L = L[:-1]
    s = (r"\begin{tabular}{@{}llcccccc@{}}" "\n" r"\toprule" "\n"
         r" & & \multicolumn{5}{c}{Pair-weighted} & Src.-wt.\\" "\n"
         r"\cmidrule(lr){3-7}\cmidrule(l){8-8}" "\n"
         r"Model & Training & $E\!\downarrow$ & $G\!\downarrow$ & Chg.$\uparrow$ & Unch.$\uparrow$"
         r" & Goal$\uparrow$ & $E\!\downarrow$ \\" "\n" r"\midrule" "\n"
         + "\n".join(L) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    _write(out, s)


# ================================================================== Figure 2 (radar)
E_MAX = 0.20   # E or G of 0.20 maps to the centre, 0 to the outer ring


def fig_radar(out):
    d = load_models()
    both = load_discrimination()
    models = [("olmotaur_base", "OLMo-2-1B", "#8E99A6", (0, (3, 1.6))),
              ("qwen3_30b_a3b_gptq", "Qwen3-30B-A3B", CORAL, "-"),
              ("socialr1_8", "SocialR1-8B", "#D79A2B", "-"),
              ("qwen3_14b", "Qwen3-14B", NAVY, "-")]
    axes_lab = ["Presentation\nstability", "Effect\nstability", "Changed\nfacts",
                "Unchanged\nfacts", "Goal\nexecution", "Both opposite\ngoals"]
    n = len(axes_lab)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    ang_c = np.concatenate([ang, ang[:1]])

    def score(m):
        r = d.loc[m]
        s = [max(0.0, 1 - r.E / E_MAX), max(0.0, 1 - r.G / E_MAX), r.changed_fact_U,
             r.stable_fact_U, r.goal_Q, both[m]]
        note("F2", m, "E=%.4f G=%.4f chg=%.4f unch=%.4f goal=%.4f both=%.4f" % (
            r.E, r.G, r.changed_fact_U, r.stable_fact_U, r.goal_Q, both[m]))
        return np.array(s)

    fig = plt.figure(figsize=(COL_W, 2.5))
    ax = fig.add_axes([0.17, 0.2, 0.66, 0.72], polar=True)
    ax.set_theta_offset(0)
    ax.set_ylim(0, 1.0)
    # rings
    for rr in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(ang_c, [rr] * (n + 1), color=GRID if rr < 1 else "#B9C2CD",
                lw=0.6 if rr < 1 else 0.9, zorder=1)
    for a in ang:
        ax.plot([a, a], [0, 1], color=GRID, lw=0.6, zorder=1)
    ax.fill(ang_c, [1] * (n + 1), color="#F3F6F9", zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    for rr in [0.5, 1.0]:
        ax.text(np.pi / 2 + np.pi / 6, rr * np.cos(np.pi / 6) + 0.02, "%d%%" % (100 * rr), fontsize=5.6,
                color="#7A8593", ha="left", va="top")
    for a, lab in zip(ang, axes_lab):
        ha = "center" if abs(np.cos(a)) < 0.2 else ("left" if np.cos(a) > 0 else "right")
        ax.text(a, 1.16, lab, ha=ha, va="center", fontsize=6.7, color=INK, linespacing=0.95)
    for m, nm, col, ls in models:
        s = score(m)
        sc = np.concatenate([s, s[:1]])
        ax.fill(ang_c, sc, color=col, alpha=0.10, zorder=2)
        ax.plot(ang_c, sc, color=col, lw=1.35, ls=ls, zorder=3, label=nm,
                solid_joinstyle="round")
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), handlelength=1.6,
               columnspacing=1.0, fontsize=6.6)
    _save(fig, out)


# ================================================================== Figure 3 (lines)
def fig_training_steps(out):
    t = _csv(P("results", "paper_tables", "mechanism_checkpoint_means.csv"))
    strata = [("train", "Training sources", NAVY, "o"),
              ("heldout_variant", "Held-out rewrites", TEAL, "s"),
              ("heldout_source", "Held-out sources", CORAL, "D")]
    models = [("qwen3_8b", "Qwen3-8B"), ("qwen25_14b_instruct", "Qwen2.5-14B")]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.45), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    lo = 100 * t.delta_decoder_W1_conditional.min()
    for ax, (m, nm) in zip(axes, models):
        ax.axhspan(-8, 0, color="#EAF3F1", zorder=0)
        ax.axhline(0, color="#55606E", lw=0.6, zorder=1)
        for st, lab, col, mk in strata:
            a = t[(t.model == m) & (t.stratum == st)].sort_values("step")
            y = 100 * a.delta_decoder_W1_conditional.values
            ax.plot(a.step, y, color=col, lw=1.5, marker=mk, ms=3.0, mec="white", mew=0.5,
                    label=lab, zorder=3)
            for s_, v in zip(a.step, y):
                note("F3", m, st, s_, "%.5f" % (v / 100))
        ax.set_title(nm, fontsize=7.2, pad=3, fontweight="bold")
        ax.set_xticks([50, 100, 150])
        ax.set_xlim(40, 160)
        ax.set_xlabel("Training steps", labelpad=1)
        ax.tick_params(axis="y", length=2.0, width=0.5)
        ax.spines["left"].set_visible(True)
        ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    axes[0].set_ylim(math.floor(lo) - 0.6, 1.0)
    axes[0].set_ylabel(r"$\Delta W_1$ ($\times 10^{-2}$)", labelpad=1.5)
    axes[0].text(45, math.floor(lo) - 0.35, "lower with consistency term", fontsize=5.8,
                 color=TEAL, style="italic", va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 1.13),
               handlelength=1.8, columnspacing=1.0, fontsize=6.6)
    _save(fig, out)


def text_numbers():
    """Numbers quoted in the RQ1 text but not plotted: the twelve instruction-tuned or
    specialized configurations with >= 85% changed-fact accuracy."""
    d = load_models()
    both = load_discrimination()
    keep = [m for m in d.index
            if m not in BASES and m not in ("qwen25_7b_base", "sotopia_base")
            and d.loc[m, "changed_fact_U"] >= 0.85]
    for m in keep:
        note("TXT", m, "chg=%.4f goal=%.4f both=%.4f" % (
            d.loc[m, "changed_fact_U"], d.loc[m, "goal_Q"], both[m]))
    note("TXT", "n=%d goal range %.4f-%.4f both range %.4f-%.4f" % (
        len(keep), min(d.loc[keep, "goal_Q"]), max(d.loc[keep, "goal_Q"]),
        min(both[m] for m in keep), max(both[m] for m in keep)))


if __name__ == "__main__":
    os.chdir(HERE)
    text_numbers()
    table_main("tables/tab_main.tex")
    table_reasoning("tables/tab_reasoning.tex")
    table_training("tables/tab_training.tex")
    # Figures 2-3: the chosen designs live in make_figure_options.py (closed frames,
    # red-blue palette); the old radar / line versions above are kept for reference only.
    import make_figure_options as fo
    fo.fig2(os.path.join(HERE, "figures", "fig2_scatter.pdf"), fo.FIG2_FINAL)
    fo.fig3_final(os.path.join(HERE, "figures", "fig3_training_bars.pdf"))
    d, both = load_models(), load_discrimination()
    for m in ["qwen3_14b", "qwen3_30b_a3b_gptq", "socialr1_8", "olmotaur_base"]:
        r = d.loc[m]
        note("F2", m, "E=%.4f chg=%.4f goal=%.4f both=%.4f" % (
            r.E, r.changed_fact_U, r.goal_Q, both[m]))
    t = _csv(P("results", "paper_tables", "mechanism_checkpoint_means.csv"))
    for _, r in t.sort_values(["model", "stratum", "step"]).iterrows():
        note("F3", r.model, r.stratum, r.step,
             "reduction=%.5f" % (-r.delta_decoder_W1_conditional))
    _write("numbers_used.txt", "\n".join(NUMBERS) + "\n")
    print("done:", len(NUMBERS), "numbers")
