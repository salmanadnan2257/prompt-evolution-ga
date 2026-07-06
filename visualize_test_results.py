from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("ERROR: pip install matplotlib numpy")
    sys.exit(1)

ROOT    = Path(__file__).parent
CMP_DIR = ROOT / "results" / "comparison"

T_COLOR = "#4C9BE8"
R_COLOR = "#5DBB7A"
E_COLOR = "#9B59B6"
B_COLOR = "#F4A92A"

OUTCOME_ORDER  = ["perfect", "partial_pass", "all_fail_early_stop", "compile_fail", "all_fail"]
OUTCOME_COLORS = {
    "perfect":             "#5DBB7A",
    "partial_pass":        "#4C9BE8",
    "all_fail_early_stop": "#F4A92A",
    "compile_fail":        "#9B59B6",
    "all_fail":            "#E8594C",
}
OUTCOME_LABELS = {
    "perfect":             "Perfect (all tests pass)",
    "partial_pass":        "Partial pass",
    "all_fail_early_stop": "All fail — early stop",
    "compile_fail":        "Compile failure",
    "all_fail":            "All fail",
}

RATING_COLORS = {
    800:  "#1f77b4",
    1000: "#ff7f0e",
    1100: "#2ca02c",
    1200: "#d62728",
    1300: "#9467bd",
    1400: "#8c564b",
}

STYLE = {
    "figure.dpi":        130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.28,
    "font.size":         12,
}


def _load_jsonl(path: Path) -> dict[str, dict]:
    entries = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return {e["problem_id"]: e for e in entries}


def _load_all() -> tuple[dict, dict, dict, dict]:
    t_path = CMP_DIR / "evolved_tournament_test_evaluations.jsonl"
    r_path = CMP_DIR / "evolved_roulette_test_evaluations.jsonl"
    e_path = CMP_DIR / "evolved_elitism_test_evaluations.jsonl"
    b_path = CMP_DIR / "baseline_test_evaluations.jsonl"

    for p in [t_path, r_path, e_path, b_path]:
        if not p.exists():
            print(f"ERROR: {p.name} not found.")
            sys.exit(1)

    t = _load_jsonl(t_path)
    r = _load_jsonl(r_path)
    e = _load_jsonl(e_path)
    b = _load_jsonl(b_path)

    # print(f"loaded: t={len(t)} r={len(r)} e={len(e)} b={len(b)}")

    assert len(t) == 200, f"tournament: {len(t)} entries (expected 200)"
    assert len(r) == 200, f"roulette: {len(r)} entries (expected 200)"
    assert len(e) == 200, f"elitism: {len(e)} entries (expected 200)"
    assert len(b) == 200, f"baseline: {len(b)} entries (expected 200)"
    assert set(t) == set(b), "tournament and baseline problem IDs don't match"
    assert set(r) == set(b), "roulette and baseline problem IDs don't match"
    assert set(e) == set(b), "elitism and baseline problem IDs don't match"

    t_mean = sum(v["score"] for v in t.values()) / 200
    r_mean = sum(v["score"] for v in r.values()) / 200
    e_mean = sum(v["score"] for v in e.values()) / 200
    b_mean = sum(v["score"] for v in b.values()) / 200

    print(f"[load] n=200 problems (all match)")
    print(f"  tournament evolved : {t_mean:.4f}")
    print(f"  roulette evolved   : {r_mean:.4f}")
    print(f"  elitism evolved    : {e_mean:.4f}")
    print(f"  baseline           : {b_mean:.4f}")

    return t, r, e, b


def plot_overall_score(t: dict, r: dict, e: dict, b: dict) -> None:
    t_mean = sum(v["score"] for v in t.values()) / 200
    r_mean = sum(v["score"] for v in r.values()) / 200
    e_mean = sum(v["score"] for v in e.values()) / 200
    b_mean = sum(v["score"] for v in b.values()) / 200

    labels = ["Tournament\nevolved", "Roulette\nevolved", "Elitism\nevolved", "Baseline\n(seed prompt 0)"]
    values = [t_mean, r_mean, e_mean, b_mean]
    colors = [T_COLOR, R_COLOR, E_COLOR, B_COLOR]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("#FAFAFA")

        bars = ax.bar(labels, values, color=colors, width=0.50, edgecolor="white", linewidth=1.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.4f}",
                    ha="center", va="bottom",
                    fontsize=12, fontweight="bold",
                    color=bar.get_facecolor())

        ax.set_ylim(0, 1.10)
        ax.set_ylabel("Mean pass-rate", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title("Overall Score on Test Set  (n=200 problems)", fontsize=12, fontweight="bold", pad=14)
        ax.grid(axis="y", alpha=0.28)

    out = CMP_DIR / "test_1_overall_score.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[1] saved: {out.name}")


def plot_outcome_distribution(t: dict, r: dict, e: dict, b: dict) -> None:
    def reason_counts(d: dict) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for v in d.values():
            counts[v["score_reason"]] += 1
        return dict(counts)

    t_out = reason_counts(t)
    r_out = reason_counts(r)
    e_out = reason_counts(e)
    b_out = reason_counts(b)

    present  = [reason for reason in OUTCOME_ORDER if any(d.get(reason, 0) > 0 for d in [t_out, r_out, e_out, b_out])]
    x_labels = ["Tournament\nevolved", "Roulette\nevolved", "Elitism\nevolved", "Baseline"]
    dists    = [t_out, r_out, e_out, b_out]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#FAFAFA")

        bottoms = np.zeros(4)
        patches = []
        for reason in present:
            counts = np.array([d.get(reason, 0) for d in dists], dtype=float)
            fracs  = counts / 200
            bars   = ax.bar(x_labels, fracs, bottom=bottoms,
                            color=OUTCOME_COLORS[reason], width=0.50, edgecolor="white", linewidth=1.0)
            for bar, frac, count, bot in zip(bars, fracs, counts, bottoms):
                if frac >= 0.04:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bot + frac / 2,
                            f"{int(count)}  ({frac*100:.0f}%)",
                            ha="center", va="center",
                            fontsize=9.5, fontweight="bold", color="white")
            bottoms += fracs
            patches.append(mpatches.Patch(color=OUTCOME_COLORS[reason], label=OUTCOME_LABELS[reason]))

        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Fraction of test problems", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title("Outcome Breakdown on Test Set  (n=200 problems)", fontsize=12, fontweight="bold", pad=14)
        ax.legend(handles=patches[::-1], loc="upper right", fontsize=9, framealpha=0.92, edgecolor="#CCCCCC")
        ax.grid(axis="y", alpha=0.28)

    out = CMP_DIR / "test_2_outcome_distribution.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[2] saved: {out.name}")


def plot_score_by_rating(t: dict, r: dict, e: dict, b: dict) -> None:
    def by_rating(d: dict) -> dict[int, list[float]]:
        groups: dict[int, list[float]] = defaultdict(list)
        for v in d.values():
            groups[v["problem_rating"]].append(v["score"])
        return groups

    t_r = by_rating(t)
    r_r = by_rating(r)
    e_r = by_rating(e)
    b_r = by_rating(b)
    ratings = sorted(t_r.keys())

    t_means = [sum(t_r[rt]) / len(t_r[rt]) for rt in ratings]
    r_means = [sum(r_r[rt]) / len(r_r[rt]) for rt in ratings]
    e_means = [sum(e_r[rt]) / len(e_r[rt]) for rt in ratings]
    b_means = [sum(b_r[rt]) / len(b_r[rt]) for rt in ratings]
    ns      = [len(b_r[rt]) for rt in ratings]

    x = np.arange(len(ratings))
    w = 0.19

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13, 7))
        fig.patch.set_facecolor("#FAFAFA")

        bars_t = ax.bar(x - 1.5*w, t_means, w, label="Tournament evolved", color=T_COLOR, edgecolor="white", linewidth=1.2, alpha=0.92)
        bars_r = ax.bar(x - 0.5*w, r_means, w, label="Roulette evolved",   color=R_COLOR, edgecolor="white", linewidth=1.2, alpha=0.92)
        bars_e = ax.bar(x + 0.5*w, e_means, w, label="Elitism evolved",    color=E_COLOR, edgecolor="white", linewidth=1.2, alpha=0.92)
        bars_b = ax.bar(x + 1.5*w, b_means, w, label="Baseline",           color=B_COLOR, edgecolor="white", linewidth=1.2, alpha=0.92)

        for bar, val in zip(bars_t, t_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#2A6FAD")
        for bar, val in zip(bars_r, r_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#2E7D4F")
        for bar, val in zip(bars_e, e_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#6C3483")
        for bar, val in zip(bars_b, b_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#C47A00")

        ax.set_xticks(x)
        ax.set_xticklabels([f"Rating {rt}\n(n={n})" for rt, n in zip(ratings, ns)], fontsize=10)
        ax.set_xlabel("Codeforces Problem Difficulty Rating", fontsize=11)
        ax.set_ylabel("Mean pass-rate", fontsize=11)
        ax.set_ylim(0, 1.30)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend(fontsize=10, framealpha=0.92, edgecolor="#CCCCCC",
                  loc="upper right", bbox_to_anchor=(1.0, 1.0))
        ax.set_title("Score by Difficulty Rating on Test Set", fontsize=12, fontweight="bold", pad=14)

    out = CMP_DIR / "test_3_score_by_rating.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[3] saved: {out.name}")


def plot_per_problem_scatter(t: dict, r: dict, e: dict, b: dict) -> None:
    pids = sorted(b.keys())

    t_scores = np.array([t[pid]["score"] for pid in pids])
    r_scores = np.array([r[pid]["score"] for pid in pids])
    e_scores = np.array([e[pid]["score"] for pid in pids])
    b_scores = np.array([b[pid]["score"] for pid in pids])
    ratings  = [b[pid]["problem_rating"] for pid in pids]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=True)
        fig.patch.set_facecolor("#FAFAFA")

        for ax, ev_scores, ev_color, ev_label in [
            (axes[0], t_scores, T_COLOR, "Tournament evolved"),
            (axes[1], r_scores, R_COLOR, "Roulette evolved"),
            (axes[2], e_scores, E_COLOR, "Elitism evolved"),
        ]:
            ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=1.2, ls="--", zorder=1, label="Equal performance")

            for rating in sorted(set(ratings)):
                mask = np.array([rt == rating for rt in ratings])
                ax.scatter(b_scores[mask], ev_scores[mask],
                           color=RATING_COLORS.get(rating, "#888"),
                           s=28, alpha=0.65, zorder=2, label=f"Rating {rating}")

            wins   = int(np.sum(ev_scores > b_scores))
            losses = int(np.sum(ev_scores < b_scores))
            ties   = 200 - wins - losses

            ax.text(0.03, 0.97, f"Evolved wins:  {wins}",  transform=ax.transAxes, fontsize=9, va="top", color=ev_color, fontweight="bold")
            ax.text(0.03, 0.91, f"Baseline wins: {losses}", transform=ax.transAxes, fontsize=9, va="top", color=B_COLOR, fontweight="bold")
            ax.text(0.03, 0.85, f"Tied:          {ties}",   transform=ax.transAxes, fontsize=9, va="top", color="#555")

            ax.set_xlim(-0.02, 1.05)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xlabel("Baseline score (per problem)", fontsize=10)
            ax.set_title(f"{ev_label}\n(above diagonal = evolved wins)", fontsize=10, fontweight="bold")
            ax.set_aspect("equal")
            ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92, edgecolor="#CCCCCC")

        axes[0].set_ylabel("Evolved score (per problem)", fontsize=10)
        fig.suptitle("Per-Problem Score vs Baseline  (n=200 test problems)", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()

    out = CMP_DIR / "test_4_per_problem_scatter.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[4] saved: {out.name}")


def plot_win_loss_tie(t: dict, r: dict, e: dict, b: dict) -> None:
    pids = sorted(b.keys())
    t_s  = [t[pid]["score"] for pid in pids]
    r_s  = [r[pid]["score"] for pid in pids]
    e_s  = [e[pid]["score"] for pid in pids]
    b_s  = [b[pid]["score"] for pid in pids]

    def wlt(ev, base):
        wins   = sum(1 for a, bv in zip(ev, base) if a > bv)
        losses = sum(1 for a, bv in zip(ev, base) if a < bv)
        return wins, 200 - wins - losses, losses

    t_w, t_tie, t_l = wlt(t_s, b_s)
    r_w, r_tie, r_l = wlt(r_s, b_s)
    e_w, e_tie, e_l = wlt(e_s, b_s)

    labels = ["Evolved wins", "Tied", "Baseline wins"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
        fig.patch.set_facecolor("#FAFAFA")

        for ax, vals, title, ev_color in [
            (axes[0], [t_w, t_tie, t_l], "Tournament evolved vs Baseline", T_COLOR),
            (axes[1], [r_w, r_tie, r_l], "Roulette evolved vs Baseline",   R_COLOR),
            (axes[2], [e_w, e_tie, e_l], "Elitism evolved vs Baseline",    E_COLOR),
        ]:
            colors = [ev_color, "#AAAAAA", B_COLOR]
            bars = ax.bar(labels, vals, color=colors, width=0.45, edgecolor="white", linewidth=1.8)
            for bar, val in zip(bars, vals):
                pct = val / 200 * 100
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8,
                        f"{val}\n({pct:.1f}%)",
                        ha="center", va="bottom",
                        fontsize=11, fontweight="bold",
                        color="#555555")
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.grid(axis="y", alpha=0.28)

        axes[0].set_ylabel("Number of problems", fontsize=11)
        axes[0].set_ylim(0, 200)
        fig.suptitle("Problem-Level Win / Tie / Loss vs Baseline  (n=200 test problems)", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()

    out = CMP_DIR / "test_5_win_loss_tie.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[5] saved: {out.name}")


def main() -> None:
    t, r, e, b = _load_all()
    print()
    plot_overall_score(t, r, e, b)
    plot_outcome_distribution(t, r, e, b)
    plot_score_by_rating(t, r, e, b)
    plot_per_problem_scatter(t, r, e, b)
    plot_win_loss_tie(t, r, e, b)
    print(f"\nAll 5 plots saved to: {CMP_DIR}")


if __name__ == "__main__":
    main()
