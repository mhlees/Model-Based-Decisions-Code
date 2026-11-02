"""
Plotting utilities for EV Stag Hunt experiments.

All functions accept pandas DataFrames produced by ev_experiments
and save figures to disk, returning the output path.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _default_plot_path(filename: str) -> str:
    plots_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return os.path.join(plots_dir, filename)


def plot_fanchart(traces_df: pd.DataFrame, out_path: Optional[str] = None) -> str:
    """Plot fan charts (quantile bands) for baseline vs subsidy using traces DF.

    traces_df columns: ['group', 'trial', 'time', 'X'] where group in {'baseline','subsidy'}.
    """
    if traces_df.empty:
        raise ValueError("traces_df is empty")

    groups = ["baseline", "subsidy"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    for j, group in enumerate(groups):
        gdf = traces_df[traces_df["group"] == group]

        # Compute quantiles by time across trials
        q = gdf.groupby("time")["X"].quantile([0.10, 0.25, 0.75, 0.90]).unstack(level=1)
        mean = gdf.groupby("time")["X"].mean()
        t = mean.index.to_numpy()

        ax = axes[0, j]
        ax.fill_between(t, q[0.10], q[0.90], color=("steelblue" if group == "baseline" else "darkorange"), alpha=0.15, label="10–90%")
        ax.fill_between(t, q[0.25], q[0.75], color=("steelblue" if group == "baseline" else "darkorange"), alpha=0.30, label="25–75%")

        # Overlay some traces for context (sample up to 100 trials)
        trial_ids = gdf["trial"].unique()
        rng = np.random.default_rng(123)
        sample = rng.choice(trial_ids, size=min(100, len(trial_ids)), replace=False)
        for tr in sample:
            tr_df = gdf[gdf["trial"] == tr]
            ax.plot(tr_df["time"], tr_df["X"], color=("steelblue" if group == "baseline" else "darkorange"), alpha=0.1, linewidth=0.8)

        ax.plot(t, mean, color=("steelblue" if group == "baseline" else "darkorange"), linewidth=2, label="mean")
        ax.set_title(f"{group.capitalize()} adoption")
        ax.set_xlabel("Time")
        ax.set_ylabel("X(t)")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")

        # Final X(T) histogram
        t_max = int(gdf["time"].max())
        final_vals = gdf[gdf["time"] == t_max].groupby("trial")["X"].mean().to_numpy()
        axes[1, j].hist(final_vals, bins=20, color=("steelblue" if group == "baseline" else "darkorange"), alpha=0.8)
        axes[1, j].set_title(f"{group.capitalize()} final X(T)")
        axes[1, j].set_xlabel("X(T)")
        axes[1, j].set_ylabel("Count")

    if out_path is None:
        out_path = _default_plot_path("ev_intervention_fanchart.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_spaghetti(traces_df: pd.DataFrame, *, max_traces: int = 100, alpha: float = 0.15, out_path: Optional[str] = None) -> str:
    """Spaghetti plot from traces DF for baseline vs subsidy."""
    groups = ["baseline", "subsidy"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    rng = np.random.default_rng(123)

    for j, group in enumerate(groups):
        gdf = traces_df[traces_df["group"] == group]
        trial_ids = gdf["trial"].unique()
        sample = rng.choice(trial_ids, size=min(max_traces, len(trial_ids)), replace=False)
        ax = axes[j]
        for tr in sample:
            tr_df = gdf[gdf["trial"] == tr]
            ax.plot(tr_df["time"], tr_df["X"], color=("steelblue" if group == "baseline" else "darkorange"), alpha=alpha, linewidth=0.8)
        ax.set_title(f"{group.capitalize()} traces")
        ax.set_xlabel("Time")
        ax.set_ylabel("X(t)")
        ax.set_ylim(0, 1)

    if out_path is None:
        out_path = _default_plot_path("ev_spaghetti.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_density(traces_df: pd.DataFrame, *, x_bins: int = 50, time_bins: Optional[int] = None, out_path: Optional[str] = None) -> str:
    """Time-evolving density plot (2D histogram) from traces DF."""
    groups = ["baseline", "subsidy"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    for j, group in enumerate(groups):
        gdf = traces_df[traces_df["group"] == group]
        T = int(gdf["time"].max()) + 1
        if time_bins is None:
            bins_time = T
        else:
            bins_time = time_bins
        hb = axes[j].hist2d(gdf["time"].to_numpy(), gdf["X"].to_numpy(), bins=[bins_time, x_bins], range=[[0, T - 1], [0.0, 1.0]], cmap="magma")
        axes[j].set_title(f"{group.capitalize()} density: time vs X(t)")
        axes[j].set_xlabel("Time")
        axes[j].set_ylabel("X(t)")
        fig.colorbar(hb[3], ax=axes[j], label="count")

    if out_path is None:
        out_path = _default_plot_path("ev_density.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_ratio_sweep(sweep_df: pd.DataFrame, out_path: Optional[str] = None) -> str:
    """Plot X* vs ratio from a DataFrame with columns ['ratio','X_mean']."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep_df["ratio"], sweep_df["X_mean"], color="C0", lw=2)
    ax.set_xlabel("a_I / b (ratio)")
    ax.set_ylabel("Final adoption X*")
    ax.set_title("X* vs ratio")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    if out_path is None:
        out_path = _default_plot_path("ev_ratio_sweep.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_phase_plot(phase_df: pd.DataFrame, out_path: Optional[str] = None) -> str:
    """Plot heatmap from tidy DataFrame with columns ['X0','ratio','X_final']."""
    # Pivot to matrix for imshow
    pivot = phase_df.pivot(index="ratio", columns="X0", values="X_final").sort_index().sort_index(axis=1)
    ratios = pivot.index.to_numpy()
    X0s = pivot.columns.to_numpy()

    plt.figure(figsize=(7, 4))
    im = plt.imshow(
        pivot.to_numpy(),
        origin="lower",
        extent=[X0s[0], X0s[-1], ratios[0], ratios[-1]],
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="plasma",
    )
    plt.colorbar(im, label="Final adopters X*")
    plt.xlabel("X0 (initial adoption)")
    plt.ylabel("a_I / b (initial payoff ratio)")
    plt.title("Network phase plot: X* over X0 and a_I/b")

    # Overlay threshold X = 1/ratio
    X_thresh = 1.0 / ratios
    X_thresh_clipped = np.clip(X_thresh, 0.0, 1.0)
    plt.plot(X_thresh_clipped, ratios, color="white", linestyle="--", linewidth=1.5, label="X = b / a_I (initial)")
    plt.legend(loc="upper right")

    if out_path is None:
        out_path = _default_plot_path("ev_phase_plot.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return out_path


# -----------------------------
# New Plotting Routines for Refactored Assignment 3
# -----------------------------

def plot_phase_X0_vs_I0(
    phase_df: pd.DataFrame,
    out_path: Optional[str] = None,
    title: str = "Figure 1: Baseline Tipping Landscape ($X^*$ over $X_0, I_0$)",
) -> str:
    """Plot 2D heatmap of final adoption X* as a function of (X0, I0).

    phase_df columns: ['X0', 'I0', 'X_final']
    """
    pivot = phase_df.pivot(index="I0", columns="X0", values="X_final").sort_index().sort_index(axis=1)
    I0s = pivot.index.to_numpy()
    X0s = pivot.columns.to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    im = ax.imshow(
        pivot.to_numpy(),
        origin="lower",
        extent=[X0s[0], X0s[-1], I0s[0], I0s[-1]],
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Final Adoption Fraction $X^*$", fontsize=11)

    # Separatrix contour line at X* = 0.5
    X_grid, I_grid = np.meshgrid(X0s, I0s)
    try:
        cs = ax.contour(X_grid, I_grid, pivot.to_numpy(), levels=[0.5], colors="red", linewidths=2.0)
        ax.clabel(cs, inline=True, fontsize=10, fmt={0.5: "Separatrix (X*=0.5)"})
    except Exception:
        pass

    ax.set_xlabel("Initial Adoption Fraction $X_0$", fontsize=12)
    ax.set_ylabel("Initial Infrastructure $I_0$", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.15, linestyle=":")

    if out_path is None:
        out_path = _default_plot_path("figure1_phase_heatmap.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_topology_comparison(
    er_X: list[np.ndarray],
    er_I: list[np.ndarray],
    ba_X: list[np.ndarray],
    ba_I: list[np.ndarray],
    out_path: Optional[str] = None,
    title: str = "Figure 2: Network Structure Comparison (ER vs. BA near Tipping)",
) -> str:
    """Plot comparative dynamic time-series traces of X(t) and I(t) for ER vs. BA."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    t = np.arange(len(er_X[0]))

    configs = [
        ("Erdős–Rényi (Random, low clustering)", er_X, er_I, axes[0], "steelblue"),
        ("Barabási–Albert (Scale-free, hubs)", ba_X, ba_I, axes[1], "darkorange"),
    ]

    for label, x_list, i_list, ax, color in configs:
        x_mat = np.vstack(x_list)
        i_mat = np.vstack(i_list)
        x_mean = x_mat.mean(axis=0)
        x_std = x_mat.std(axis=0)

        # Plot individual trajectories (thin)
        for X in x_list:
            ax.plot(t, X, color=color, alpha=0.25, linewidth=1.0)

        # Plot mean trajectory with +/- 1 std
        ax.plot(t, x_mean, color=color, linewidth=2.5, label="Mean $X(t)$")
        ax.fill_between(t, np.clip(x_mean - x_std, 0, 1), np.clip(x_mean + x_std, 0, 1), color=color, alpha=0.2, label="$\pm 1$ Std Dev")

        # Mean infrastructure
        i_mean = i_mat.mean(axis=0)
        ax.plot(t, i_mean, color="forestgreen", linestyle="--", linewidth=2.0, label="Mean Infrastructure $I(t)$")

        success_rate = np.mean([X[-1] > 0.5 for X in x_list]) * 100
        ax.set_title(f"{label}\nSuccess Rate: {success_rate:.0f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time Step $t$", fontsize=11)
        ax.set_ylabel("Adoption Fraction $X(t)$", fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(loc="center right", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    if out_path is None:
        out_path = _default_plot_path("figure2_topology_comparison.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_intervention_comparison(
    baseline_X: list[np.ndarray],
    intervention_X: list[np.ndarray],
    policy_label: str = "Intervention",
    timing_window: Optional[tuple[int, int]] = None,
    out_path: Optional[str] = None,
    title: str = "Figure 3: Policy Intervention vs. Baseline",
) -> str:
    """Plot baseline vs. intervention comparison showing stochastic trajectories and mean."""
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    t = np.arange(len(baseline_X[0]))

    b_mat = np.vstack(baseline_X)
    i_mat = np.vstack(intervention_X)

    # Plot baseline
    for X in baseline_X:
        ax.plot(t, X, color="slategray", alpha=0.2, linewidth=1.0)
    ax.plot(t, b_mat.mean(axis=0), color="dimgray", linewidth=2.5, label=f"Baseline (Unforced, mean final={b_mat[:, -1].mean():.2f})")

    # Plot intervention
    for X in intervention_X:
        ax.plot(t, X, color="crimson", alpha=0.2, linewidth=1.0)
    ax.plot(t, i_mat.mean(axis=0), color="crimson", linewidth=2.5, label=f"{policy_label} (mean final={i_mat[:, -1].mean():.2f})")

    if timing_window is not None:
        ax.axvspan(timing_window[0], timing_window[1], color="gold", alpha=0.2, label=f"Active Window [{timing_window[0]}, {timing_window[1]}]")

    ax.set_xlabel("Time Step $t$", fontsize=12)
    ax.set_ylabel("Adoption Fraction $X(t)$", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="center right", fontsize=10)

    if out_path is None:
        out_path = _default_plot_path("figure3_policy_evaluation.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path