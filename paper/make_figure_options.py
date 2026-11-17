# -*- coding: utf-8 -*-
"""Style variants of the chosen Figure 2 (scatter map) and Figure 3 (upright bars).

Reads the same accepted tables as make_figures.py; nothing is typed by hand.
main.tex is untouched.  Run:  python make_figure_options.py
Outputs: figures/options/fig2_v*.pdf|png, figures/options/fig3_v*.pdf|png, overview_*.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures as mf                                    # noqa: E402  (fonts, rcParams)
from make_figures import (COL_W, INK, P, _csv, SPECIALIZED, BASES,   # noqa: E402
                          load_models, load_discrimination)

OUT = os.path.join(HERE, "figures", "options")

# ---------------------------------------------------------------- palette (user-supplied)
PAL = ["#104680", "#317CB7", "#6DADD1", "#B6D7E8", "#E9F1F4",
       "#FBE3D5", "#F6B293", "#DC6D57", "#B72230", "#6D011F"]
B0, B1, B2, B3, B4, R5, R6, R7, R8, R9 = PAL
FRAME = "#2E3338"
MUTED = "#6B7480"
GRIDC = "#E3E7EC"

plt.rcParams.update({
    "axes.spines.top": True, "axes.spines.right": True,      # closed frame everywhere
    "axes.edgecolor": FRAME, "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.color": FRAME, "ytick.color": FRAME,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "hatch.linewidth": 0.45,
    "legend.fontsize": 6.2,
})


def save(fig, name):
    """name: option name (saved under figures/options) or a full path ending in .pdf."""
    base = name[:-4] if name.endswith(".pdf") else os.path.join(OUT, name)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=300)
    plt.close(fig)
    print("saved", name)


def framed_legend(leg):
    f = leg.get_frame()
    f.set_edgecolor(FRAME)
    f.set_linewidth(0.5)
    f.set_facecolor("white")
    f.set_alpha(1)
    f.set_boxstyle("square", pad=0.25)
    return leg


# ================================================================ Figure 2 (scatter map)
FAMS = [("Instruction-tuned", B0, "o"), ("Social reasoning", R8, "s"),
        ("Behavior adapters", R6, "^"), ("Bases / controls", B3, "o")]


def family(m):
    if m in SPECIALIZED:
        return "Social reasoning" if m.startswith("soci") else "Behavior adapters"
    if m in BASES or m in ("qwen25_7b_base", "sotopia_base"):
        return "Bases / controls"
    return "Instruction-tuned"


# label anchor positions (data coordinates) chosen to sit in empty space
LAB_L = {"qwen3_14b": ("Qwen3-14B", 0.060, 90.5, "right"),
         "qwen3_30b_a3b_gptq": ("Qwen3-30B-A3B", 0.285, 83.0, "left"),
         "socialr1_8": ("SocialR1-8B", 0.165, 51.5, "center"),
         "olmotaur_base": ("OLMo-2-1B", 0.0102, 58.5, "right")}
LAB_R = {"qwen3_14b": ("Qwen3-14B", 74.0, 94.0, "center"),
         "qwen3_30b_a3b_gptq": ("Qwen3-30B-A3B", 67.5, 86.0, "center"),
         "socialr1_8": ("SocialR1-8B", 100.5, 47.8, "right"),
         "olmotaur_base": ("OLMo-2-1B", 45.6, 68.0, "left")}


def fig2(name, st):
    """st keys: marker ('white'|'black'|'hollow'|'cmap'), grid, bg, ticks_in, region,
    tint, legend ('top'|'inside')."""
    d = load_models()
    both = load_discrimination()
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, st.get("h", 2.05)), sharey=True,
                               gridspec_kw={"wspace": 0.07})
    cmap = LinearSegmentedColormap.from_list("rdbu", PAL[::-1])
    for ax in (a, b):
        if st.get("bg"):
            ax.set_facecolor(st["bg"])
        if st.get("grid"):
            ax.grid(True, color=st["grid"], lw=0.5 if not st.get("bg") else 0.7,
                    ls=st.get("grid_ls", "-"), zorder=0)
            ax.set_axisbelow(True)
        if st.get("ticks_in"):
            ax.tick_params(direction="in", top=True, right=True, which="both")
        else:
            ax.tick_params(which="both", top=False, right=False)
    # ------------------------------------------------ points
    sc = None
    for fam, col, mk in FAMS:
        ms = [m for m in d.index if family(m) == fam]
        xs_l, ys = d.loc[ms, "E"], 100 * d.loc[ms, "goal_Q"]
        xs_r = 100 * d.loc[ms, "changed_fact_U"]
        mode = st["marker"]
        if mode == "cmap":
            cv = [100 * both[m] for m in ms]
            kw = dict(c=cv, cmap=cmap, vmin=0, vmax=55, marker=mk, s=20,
                      edgecolor=FRAME, linewidth=0.35, zorder=3)
            sc = a.scatter(xs_l, ys, **kw)
            b.scatter(xs_r, ys, **kw)
            a.scatter([], [], marker=mk, s=16, color="#9AA3AD", edgecolor=FRAME,
                      linewidth=0.35, label=fam)
            continue
        if mode == "white":
            kw = dict(color=col, edgecolor="white", linewidth=0.5, s=20)
        elif mode == "black":
            kw = dict(color=col, edgecolor=FRAME, linewidth=0.4, s=17)
        else:   # hollow
            kw = dict(facecolor="white", edgecolor=col if col != B3 else B2,
                      linewidth=0.85, s=17)
        a.scatter(xs_l, ys, marker=mk, zorder=3, label=fam, **kw)
        b.scatter(xs_r, ys, marker=mk, zorder=3, **kw)
    # highlighted four: thin ring + leader label
    for ax, lab, xcol, xmul in [(a, LAB_L, "E", 1), (b, LAB_R, "changed_fact_U", 100)]:
        for m, (nm, tx, ty, ha) in lab.items():
            px, py = xmul * d.loc[m, xcol], 100 * d.loc[m, "goal_Q"]
            if st["marker"] == "hollow":
                col = dict((f, c) for f, c, _ in FAMS)[family(m)]
                ax.scatter(px, py, marker=dict((f, k) for f, _, k in FAMS)[family(m)],
                           s=17, color=col if col != B3 else B2, edgecolor="white",
                           linewidth=0.4, zorder=4)
            ax.scatter(px, py, s=46, facecolor="none", edgecolor=INK, linewidth=0.5,
                       zorder=5)
            ax.annotate(nm, (px, py), xytext=(tx, ty), textcoords="data", ha=ha,
                        va="center", fontsize=5.9, color=INK, zorder=6,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.45,
                                        shrinkA=1.2, shrinkB=3.2))
    # ------------------------------------------------ left panel: stability vs goal
    a.set_xscale("log")
    a.set_xlim(0.30, 0.008)
    a.set_xticks([0.2, 0.1, 0.05, 0.02, 0.01])
    a.set_xticklabels([".20", ".10", ".05", ".02", ".01"])
    a.minorticks_off()
    a.set_ylim(44, 100)
    a.set_yticks([50, 60, 70, 80, 90, 100])
    a.set_xlabel(r"Sensitivity $E$ ($\rightarrow$ more stable)", labelpad=1.5, fontsize=6.6)
    a.set_ylabel("Goal accuracy (%)", labelpad=1.5)
    if st.get("region", True):
        a.add_patch(Rectangle((0.008, 80), 0.05 - 0.008, 20, facecolor=B4,
                              edgecolor=B2, lw=0.6, ls=(0, (3, 1.5)), zorder=1))
        a.text(0.02, 90, "stable and\ngoal-correct:\nnone", ha="center", va="center",
               fontsize=5.5, color=B0, style="italic", linespacing=0.95, zorder=2)
    # ------------------------------------------------ right panel: fact vs goal
    b.set_xlim(44, 102)
    b.set_xticks([50, 60, 70, 80, 90, 100])
    b.set_xlabel("Changed-fact accuracy (%)", labelpad=1.5, fontsize=6.6)
    if st.get("tint", True):
        b.fill_between([44, 102], [44, 102], [44, 44], color=R5, alpha=0.75, zorder=0,
                       lw=0)
    b.plot([44, 102], [44, 102], color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=2)
    b.text(85.5, 87.0, "goal = fact", rotation=45, transform_rotates_text=True,
           rotation_mode="anchor", fontsize=5.5, color=MUTED, ha="left", va="bottom")
    b.text(63.5, 61.0, "knows the change,\nmisses the goal", rotation=45,
           transform_rotates_text=True, rotation_mode="anchor", fontsize=5.3, color=R8,
           style="italic", ha="left", va="top", linespacing=0.95)
    # ------------------------------------------------ legend / colorbar
    h, l = a.get_legend_handles_labels()
    if st.get("legend", "top") == "top":
        leg = fig.legend(h, l, loc="lower center", ncol=4,
                         bbox_to_anchor=(0.52, a.get_position().y1 + (0.035 if st.get("legend_frame") else 0.012)),
                         fontsize=5.9, handletextpad=0.05, columnspacing=0.75,
                         borderaxespad=0, frameon=st.get("legend_frame", False))
        if st.get("legend_frame"):
            framed_legend(leg)
        if sc is None:
            # widen the two panels so their outer frames line up with the legend's ends
            fig.canvas.draw()
            bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
            left = a.get_position().x0
            fig.subplots_adjust(left=left, right=left + bb.width)
            leg.set_bbox_to_anchor((left + bb.width / 2, a.get_position().y1 + 0.012),
                                   transform=fig.transFigure)
    if sc is not None:
        cb = fig.colorbar(sc, ax=[a, b], fraction=0.035, pad=0.02, aspect=18)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(length=1.8, width=0.4, labelsize=5.6, pad=1.2)
        cb.set_ticks([0, 25, 50])
        cb.set_label("Both opposite goals (%)", fontsize=5.8, labelpad=1.5)
    save(fig, name)


# ================================================================ Figure 3 (upright bars)
STRATA = [("train", "Training sources"), ("heldout_variant", "Held-out rewrites"),
          ("heldout_source", "Held-out sources")]
MODELS = [("qwen3_8b", "Qwen3-8B"), ("qwen25_14b_instruct", "Qwen2.5-14B")]
STEPS = [50, 100, 150]
DEC = "delta_decoder_W1_conditional"
YLAB = r"Reduction in $W_1$ ($\times 10^{-2}$)"


def red(t, m, st):
    """Reduction (SFT minus SFT+consistency) at 50/100/150 steps, x100; positive = helps."""
    a = t[(t.model == m) & (t.stratum == st)].sort_values("step")
    assert list(a.step) == STEPS
    return -100 * a[DEC].values


def fmt(v):
    return ("%.1f" % v).replace("-", "−")


def bars_axes(h=1.62, sharey=True):
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, h), sharey=sharey,
                             gridspec_kw={"wspace": 0.06})
    for ax in axes:
        ax.axhline(0, color=FRAME, lw=0.5, zorder=2)
        ax.tick_params(axis="x", length=0, top=False)
        ax.tick_params(axis="y", right=False)
    return fig, axes


def fig3_grouped(name, colors, edge=None, hatches=None, labels=False, legend="inside",
                 grid=True, title_inside=True, light=False):
    t = _csv(P("results", "paper_tables", "mechanism_checkpoint_means.csv"))
    fig, axes = bars_axes(1.62 if legend == "inside" else 1.7)
    w = 0.27
    for ax, (m, nm) in zip(axes, MODELS):
        if grid:
            ax.yaxis.grid(True, color=GRIDC, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
        for k, (st, lab) in enumerate(STRATA):
            y = red(t, m, st)
            xs = np.arange(3) + (k - 1) * w
            fc = colors[k]
            ec = edge[k] if isinstance(edge, (list, tuple)) else (edge or fc)
            ax.bar(xs, y, w * 0.86, color=fc, edgecolor=ec, linewidth=0.45 if edge else 0,
                   hatch=hatches[k] if hatches else None, label=lab, zorder=3)
            if labels:
                for xx, v in zip(xs, y):
                    ax.text(xx, max(v, 0) + 0.12, fmt(v), ha="center", va="bottom",
                            fontsize=4.6, color=INK, zorder=4)
        ax.set_xticks(range(3))
        ax.set_xticklabels(STEPS)
        ax.set_xlim(-0.5, 2.5)
        ax.set_xlabel("Training steps", labelpad=1.5)
        if title_inside:
            ax.text(0.04, 0.95, nm, transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.8, fontweight="bold", color=INK)
        else:
            ax.set_title(nm, fontsize=7.0, fontweight="bold", pad=3)
    axes[0].set_ylim(-0.7, 7.6 if labels else 7.3)
    axes[0].set_yticks([0, 2, 4, 6])
    axes[0].set_ylabel(YLAB, labelpad=1.5, fontsize=6.6)
    h, l = axes[0].get_legend_handles_labels()
    if legend == "inside":
        leg = axes[0].legend(h, l, loc="upper left", bbox_to_anchor=(0.03, 0.84),
                             fontsize=5.6, handlelength=1.2, handleheight=0.75,
                             handletextpad=0.35, labelspacing=0.25, borderaxespad=0,
                             frameon=True)
        framed_legend(leg)
    else:
        fig.legend(h, l, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.53, axes[0].get_position().y1 + 0.085),
                   fontsize=6.2, handlelength=1.1, handleheight=0.75, columnspacing=1.0,
                   borderaxespad=0)
    save(fig, name)


def fig3_final(name):
    """Chosen design (v1) with value labels, panel letters, step separators and a note."""
    t = _csv(P("results", "paper_tables", "mechanism_checkpoint_means.csv"))
    fig, axes = bars_axes(1.72)
    colors = [B0, B2, R8]
    w = 0.27
    for i, (ax, (m, nm)) in enumerate(zip(axes, MODELS)):
        ax.yaxis.grid(True, color=GRIDC, lw=0.5, ls=(0, (1, 1.5)), zorder=0)
        ax.set_axisbelow(True)
        for xs in (0.5, 1.5):
            ax.axvline(xs, color="#C3CAD2", lw=0.45, ls=(0, (2, 1.5)), zorder=1)
        for k, (st, lab) in enumerate(STRATA):
            y = red(t, m, st)
            xs = np.arange(3) + (k - 1) * w
            ax.bar(xs, y, w * 0.86, color=colors[k], edgecolor=FRAME, linewidth=0.45,
                   label=lab, zorder=3)
            for xx, v in zip(xs, y):
                ax.text(xx, max(v, 0) + 0.14, fmt(v), ha="center", va="bottom",
                        fontsize=4.9, color=INK, zorder=4)
        ax.set_xticks(range(3))
        ax.set_xticklabels(STEPS)
        ax.set_xlim(-0.5, 2.5)
        ax.set_xlabel("Training steps", labelpad=1.5)
        ax.text(0.035, 0.955, "(%s) %s" % ("ab"[i], nm), transform=ax.transAxes, ha="left",
                va="top", fontsize=6.7, fontweight="bold", color=INK)
    axes[0].set_ylim(-0.7, 7.7)
    axes[0].set_yticks([0, 2, 4, 6])
    axes[0].set_ylabel(YLAB, labelpad=1.5, fontsize=6.6)
    # note: held-out sources of Qwen2.5-14B show no reduction
    v = red(t, "qwen25_14b_instruct", "heldout_source")
    axes[1].annotate("no reduction on\nheld-out sources", xy=(2 + w, max(v[2], 0) + 0.75),
                     xytext=(1.86, 5.05), ha="center", va="center", fontsize=5.4, color=R8,
                     style="italic", linespacing=0.95,
                     arrowprops=dict(arrowstyle="-|>", mutation_scale=4.5, color=R8, lw=0.5,
                                     connectionstyle="arc3,rad=-0.25", shrinkA=1, shrinkB=1))
    axes[1].text(0.035, 0.835, r"higher $=$ larger gain", transform=axes[1].transAxes,
                 ha="left", va="top", fontsize=5.3, color=MUTED, style="italic")
    h, l = axes[0].get_legend_handles_labels()
    leg = axes[0].legend(h, l, loc="upper left", bbox_to_anchor=(0.03, 0.835), fontsize=5.6,
                         handlelength=1.2, handleheight=0.75, handletextpad=0.35,
                         labelspacing=0.25, borderaxespad=0, frameon=True)
    framed_legend(leg)
    save(fig, name)


FIG2_FINAL = dict(marker="black", ticks_in=True, grid=GRIDC, grid_ls=(0, (1, 1.5)), region=False)


def fig3_by_stratum(name):
    """x = distance from training; bars = steps in a sequential blue."""
    t = _csv(P("results", "paper_tables", "mechanism_checkpoint_means.csv"))
    fig, axes = bars_axes(1.62)
    cols = [B3, B2, B0]
    w = 0.27
    for ax, (m, nm) in zip(axes, MODELS):
        ax.yaxis.grid(True, color=GRIDC, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        M = np.array([red(t, m, st) for st, _ in STRATA])     # strata x steps
        for j, s in enumerate(STEPS):
            ax.bar(np.arange(3) + (j - 1) * w, M[:, j], w * 0.86, color=cols[j],
                   edgecolor=FRAME, linewidth=0.4, label="%d steps" % s, zorder=3)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Training\nsources", "Held-out\nrewrites", "Held-out\nsources"],
                           fontsize=6.0, linespacing=0.95)
        ax.set_xlim(-0.5, 2.5)
        ax.text(0.96, 0.95, nm, transform=ax.transAxes, ha="right", va="top",
                fontsize=6.8, fontweight="bold", color=INK)
    axes[0].set_ylim(-0.7, 7.3)
    axes[0].set_yticks([0, 2, 4, 6])
    axes[0].set_ylabel(YLAB, labelpad=1.5, fontsize=6.6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3,
               bbox_to_anchor=(0.53, axes[0].get_position().y1 + 0.012), fontsize=6.2,
               handlelength=1.1, handleheight=0.75, columnspacing=1.0, borderaxespad=0)
    save(fig, name)


def fig3_horizontal(name):
    t = _csv(P("results", "paper_tables", "mechanism_checkpoint_means.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.75), sharey=True, sharex=True,
                             gridspec_kw={"wspace": 0.06})
    cols = [B0, B2, R8]
    h_ = 0.26
    for ax, (m, nm) in zip(axes, MODELS):
        ax.axvline(0, color=FRAME, lw=0.5, zorder=2)
        ax.xaxis.grid(True, color=GRIDC, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for k, (st, lab) in enumerate(STRATA):
            y = red(t, m, st)
            ys = np.arange(3) + (k - 1) * h_
            ax.barh(ys, y, h_ * 0.86, color=cols[k], edgecolor=FRAME, linewidth=0.4,
                    label=lab, zorder=3)
        ax.set_yticks(range(3))
        ax.set_yticklabels(["%d" % s for s in STEPS])
        ax.set_ylim(2.55, -0.55)
        ax.tick_params(axis="y", length=0, right=False)
        ax.tick_params(axis="x", top=False)
        ax.set_title(nm, fontsize=7.0, fontweight="bold", pad=3)
    axes[0].set_xlim(-0.7, 7.3)
    axes[0].set_xticks([0, 2, 4, 6])
    axes[0].set_ylabel("Training steps", labelpad=2)
    fig.supxlabel(YLAB, fontsize=6.6, y=-0.06)
    h, l = axes[0].get_legend_handles_labels()
    leg = axes[0].legend(h, l, loc="center right", fontsize=5.6, handlelength=1.2,
                         handleheight=0.75, handletextpad=0.35, labelspacing=0.25,
                         borderaxespad=0.35, frameon=True)
    framed_legend(leg)
    save(fig, name)


def fig3_single(name):
    """One panel: x = distance from training; blues = Qwen3-8B, reds = Qwen2.5-14B."""
    t = _csv(P("results", "paper_tables", "mechanism_checkpoint_means.csv"))
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    ax.axhline(0, color=FRAME, lw=0.5, zorder=2)
    ax.yaxis.grid(True, color=GRIDC, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    shades = {"qwen3_8b": [B3, B2, B0], "qwen25_14b_instruct": [R5, R6, R8]}
    w = 0.12
    handles = {}
    for i, (m, nm) in enumerate(MODELS):
        M = np.array([red(t, m, st) for st, _ in STRATA])
        for j, s in enumerate(STEPS):
            off = (i * 3 + j - 2.5) * w + (0.03 if i else -0.03)
            bb = ax.bar(np.arange(3) + off, M[:, j], w * 0.9, color=shades[m][j],
                        edgecolor=FRAME, linewidth=0.4, zorder=3)
            handles[(m, s)] = (bb, "%s, %d" % (nm, s))
    order = [(m, s) for s in STEPS for m, _ in MODELS]
    ax.legend([handles[k][0] for k in order], [handles[k][1] for k in order], ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.8, handlelength=1.1,
              handleheight=0.75, columnspacing=0.9, handletextpad=0.35, labelspacing=0.2,
              borderaxespad=0.2, frameon=False)
    ax.set_xticks(range(3))
    ax.set_xticklabels([s[1] for s in STRATA])
    ax.tick_params(axis="x", length=0, top=False)
    ax.tick_params(axis="y", right=False)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.7, 7.3)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_ylabel(YLAB, labelpad=1.5, fontsize=6.6)
    save(fig, name)


# ================================================================ overview sheets
def overview(prefix, items, out, ncol=3):
    plt.rcParams["font.family"] = "Microsoft YaHei"
    n = len(items)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.9 * nrow))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (f, title) in zip(axes.ravel(), items):
        ax.imshow(mpimg.imread(os.path.join(OUT, f + ".png")))
        ax.set_title(title, fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out), dpi=110, facecolor="white")
    plt.close(fig)
    plt.rcParams["font.family"] = "Linux Biolinum O"


if __name__ == "__main__":
    os.chdir(HERE)
    F2 = [("fig2_v1_clean", "v1 简洁：白边实心点 + 浅网格",
           dict(marker="white", grid=GRIDC)),
          ("fig2_v2_classic", "v2 经典期刊：黑边点、四周朝内刻度、无网格",
           dict(marker="black", ticks_in=True)),
          ("fig2_v3_colormap", "v3 颜色 = 两个相反目标都对的比例",
           dict(marker="cmap", grid=GRIDC, tint=False, h=2.1)),
          ("fig2_v4_panel", "v4 浅蓝底板 + 白网格",
           dict(marker="white", grid="white", bg=B4, region=False, tint=False)),
          ("fig2_v5_hollow", "v5 空心点（重点四个实心）",
           dict(marker="hollow", grid=GRIDC, grid_ls=(0, (1, 1.5))))]
    for name, _, st in F2:
        fig2(name, st)

    F3 = [("fig3_v1_classic", "v1 经典：深蓝/浅蓝/红 + 黑边，图例在框内",
           lambda n: fig3_grouped(n, [B0, B2, R8], edge=FRAME)),
          ("fig3_v2_labeled", "v2 柱顶标数值，图例在上方",
           lambda n: fig3_grouped(n, [B0, B2, R8], labels=True, legend="top",
                                  title_inside=False)),
          ("fig3_v3_by_stratum", "v3 按离训练远近分组，深浅 = 训练步数",
           fig3_by_stratum),
          ("fig3_v4_hatched", "v4 浅色填充 + 纹理（黑白打印友好）",
           lambda n: fig3_grouped(n, [B3, "white", R5], edge=[B0, B1, R8],
                                  hatches=["", "//////", "......"])),
          ("fig3_v5_horizontal", "v5 横向柱状图",
           fig3_horizontal),
          ("fig3_v6_single", "v6 单图合并两个模型（蓝/红）",
           fig3_single)]
    for name, _, fn in F3:
        fn(name)

    fig2("fig2_final", FIG2_FINAL)
    fig3_final("fig3_final")
    overview("fig2", [(n, t) for n, t, _ in F2], "overview_fig2.png")
    overview("fig3", [(n, t) for n, t, _ in F3], "overview_fig3.png")
