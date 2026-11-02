"""
Experiment utilities for the EV Stag Hunt model.

Contains policy factory functions, trial runners, multiprocessing helpers,
and standalone plotting routines. Depends on `ev_core` for the model.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from ev_core import (
    EVStagHuntModel,
    set_initial_adopters,
    final_mean_adoption_vs_ratio,
    phase_sweep_X0_vs_ratio,
    phase_sweep_X0_vs_I0,
    run_simulation_trial,
)
from ev_plotting import (
    plot_fanchart,
    plot_spaghetti,
    plot_density,
    plot_ratio_sweep,
    plot_phase_plot,
    plot_phase_X0_vs_I0,
    plot_topology_comparison,
    plot_intervention_comparison,
)


# -----------------------------
# Policy factories
# -----------------------------

def policy_subsidy_factory(start: int, end: int, delta_a0: float = 0.3, delta_beta_I: float = 0.0) -> Callable:
    """Create a policy that temporarily boosts coordination payoffs.

    Raises `a0` and/or `beta_I` during `[start, end)` and reverts after.
    Returns a closure `policy(model, step)`.
    """

    def policy(model, step):
        if not hasattr(policy, "base_a0"):
            policy.base_a0 = model.a0
        if not hasattr(policy, "base_beta_I"):
            policy.base_beta_I = model.beta_I

        if start <= step < end:
            model.a0 = policy.base_a0 + delta_a0
            model.beta_I = policy.base_beta_I + delta_beta_I
        else:
            model.a0 = policy.base_a0
            model.beta_I = policy.base_beta_I

    return policy


def policy_infrastructure_boost_factory(start: int, boost: float = 0.2, once: bool = True) -> Callable:
    """Create a policy that injects infrastructure at a specific step."""

    def policy(model, step):
        if step < start:
            return
        if once:
            if not hasattr(policy, "done"):
                model.infrastructure = float(np.clip(model.infrastructure + boost, 0.0, 1.0))
                policy.done = True
        else:
            model.infrastructure = float(np.clip(model.infrastructure + boost, 0.0, 1.0))

    return policy


# -----------------------------
# Trial runner
# -----------------------------

def run_timeseries_trial(
    T: int = 200,
    scenario_kwargs: Optional[Dict] = None,
    seed: Optional[int] = None,
    policy: Optional[Callable] = None,
    strategy_choice_func: str = "imitate",
    tau: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Run a single simulation and return X(t), I(t), and the model dataframe."""

    scenario = {
        # Either provide `ratio` to pin the initial a_I/b, or explicit `a0`.
        # Defaults here mirror the classroom-friendly values.
        # If `ratio` is present, we compute `a0 = ratio*b - beta_I*I0`.
        "a0": 2.0,
        "ratio": None,
        "beta_I": 3.0,
        "b": 1.0,
        "g_I": 0.1,
        "I0": 0.05,
        "network_type": "random",
        "n_nodes": 100,
        "p": 0.05,
        "m": 2,
        "collect": True,
        "X0_frac": 0.0,
        "init_method": "random",
    }
    if scenario_kwargs:
        scenario.update(scenario_kwargs)

    # Compute a0 from ratio if provided to preserve initial payoff ratio
    a0_for_model = scenario["a0"]
    if scenario.get("ratio") is not None:
        a0_for_model = float(scenario["ratio"]) * float(scenario["b"]) - float(scenario["beta_I"]) * float(scenario["I0"])

    model = EVStagHuntModel(
        a0=a0_for_model,
        beta_I=scenario["beta_I"],
        b=scenario["b"],
        g_I=scenario["g_I"],
        I0=scenario["I0"],
        seed=seed,
        network_type=scenario["network_type"],
        n_nodes=scenario["n_nodes"],
        p=scenario["p"],
        m=scenario["m"],
        collect=True,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )

    if scenario.get("X0_frac", 0.0) > 0.0:
        set_initial_adopters(
            model,
            scenario["X0_frac"],
            method=scenario.get("init_method", "random"),
            seed=seed,
            high=scenario.get("high", True),
        )

    for t in range(T):
        if policy is not None:
            policy(model, t)
        model.step()

    df = model.datacollector.get_model_vars_dataframe().copy()
    return df["X"].to_numpy(), df["I"].to_numpy(), df


def _timeseries_trial_worker(args_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Worker for parallel trials that reconstructs closures for policies."""
    T = args_dict["T"]
    scenario_kwargs = args_dict.get("scenario_kwargs", {})
    seed = args_dict.get("seed", None)
    policy_spec = args_dict.get("policy", None)
    strategy_choice_func = args_dict.get("strategy_choice_func", "imitate")
    tau = args_dict.get("tau", 1.0)

    policy = None
    if isinstance(policy_spec, dict):
        ptype = policy_spec.get("type")
        if ptype == "subsidy":
            policy = policy_subsidy_factory(**policy_spec["params"])
        elif ptype == "infrastructure":
            policy = policy_infrastructure_boost_factory(**policy_spec["params"])

    X, I, _df = run_timeseries_trial(
        T=T,
        scenario_kwargs=scenario_kwargs,
        seed=seed,
        policy=policy,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )
    return X, I


# -----------------------------
# Experiment: Intervention trials + plotting
# -----------------------------

def collect_intervention_trials(
    n_trials: int = 10,
    T: int = 200,
    scenario_kwargs: Optional[Dict] = None,
    subsidy_params: Optional[Dict] = None,
    max_workers: int = 1,
    seed_base: int = 42,
    strategy_choice_func: str = "imitate",
    tau: float = 1.0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Run baseline and subsidy trials; return raw trajectories and summary dataframes."""

    scenario = scenario_kwargs or {}
    subsidy = subsidy_params or {"start": 30, "end": 80, "delta_a0": 0.3, "delta_beta_I": 0.0}

    baseline_args = []
    subsidy_args = []
    for i in range(n_trials):
        seed = seed_base + i
        baseline_args.append(
            {
                "T": T,
                "scenario_kwargs": scenario,
                "seed": seed,
                "policy": None,
                "strategy_choice_func": strategy_choice_func,
                "tau": tau,
            }
        )
        subsidy_args.append(
            {
                "T": T,
                "scenario_kwargs": scenario,
                "seed": seed,
                "policy": {"type": "subsidy", "params": subsidy},
                "strategy_choice_func": strategy_choice_func,
                "tau": tau,
            }
        )

    baseline_X, baseline_I = [], []
    subsidy_X, subsidy_I = [], []

    # Run sequentially or concurrently
    Executor = ThreadPoolExecutor if max_workers == 1 else ProcessPoolExecutor
    with Executor(max_workers=max_workers) as ex:
        baseline_futs = [ex.submit(_timeseries_trial_worker, args) for args in baseline_args]
        subsidy_futs = [ex.submit(_timeseries_trial_worker, args) for args in subsidy_args]
        for fut in as_completed(baseline_futs):
            X, I = fut.result()
            baseline_X.append(X)
            baseline_I.append(I)
        for fut in as_completed(subsidy_futs):
            X, I = fut.result()
            subsidy_X.append(X)
            subsidy_I.append(I)

    # Align order by seed (as_completed may scramble)
    baseline_X = sorted(baseline_X, key=lambda arr: tuple(arr))
    subsidy_X = sorted(subsidy_X, key=lambda arr: tuple(arr))

    # Summary stats
    def summarize(X_list: List[np.ndarray]) -> pd.DataFrame:
        mat = np.vstack(X_list)
        df = pd.DataFrame({
            "X_mean": mat.mean(axis=0),
            "X_med": np.median(mat, axis=0),
            "X_q10": np.quantile(mat, 0.10, axis=0),
            "X_q25": np.quantile(mat, 0.25, axis=0),
            "X_q75": np.quantile(mat, 0.75, axis=0),
            "X_q90": np.quantile(mat, 0.90, axis=0),
        })
        return df

    baseline_df = summarize(baseline_X)
    subsidy_df = summarize(subsidy_X)

    return baseline_X, baseline_I, subsidy_X, subsidy_I, baseline_df, subsidy_df


def traces_to_long_df(baseline_X: List[np.ndarray], subsidy_X: List[np.ndarray]) -> pd.DataFrame:
    """Convert trajectory lists to a tidy DataFrame: [group, trial, time, X]."""
    rows = []
    for trial, X in enumerate(baseline_X):
        for t, x in enumerate(X):
            rows.append(("baseline", trial, t, float(x)))
    for trial, X in enumerate(subsidy_X):
        for t, x in enumerate(X):
            rows.append(("subsidy", trial, t, float(x)))
    return pd.DataFrame(rows, columns=["group", "trial", "time", "X"])


def ratio_sweep_df(
    X0_frac: float = 0.40,
    ratio_values: Optional[np.ndarray] = None,
    scenario_kwargs: Optional[Dict] = None,
    T: int = 250,
    batch_size: int = 16,
    init_noise_I: float = 0.04,
    strategy_choice_func: str = "logit",
    tau: float = 1.0,
) -> pd.DataFrame:
    """Compute X* vs ratio and return as a DataFrame."""
    scenario = {
        "beta_I": 2.0,
        "b": 1.0,
        "g_I": 0.05,
        "I0": 0.05,
        "network_type": "BA",
        "n_nodes": 120,
        "p": 0.05,
        "m": 2,
    }
    if scenario_kwargs:
        scenario.update(scenario_kwargs)

    if ratio_values is None:
        ratio_values = np.linspace(0.8, 3.5, 41)

    X_means = final_mean_adoption_vs_ratio(
        X0_frac,
        ratio_values,
        I0=scenario["I0"],
        beta_I=scenario["beta_I"],
        b=scenario["b"],
        g_I=scenario["g_I"],
        T=T,
        network_type=scenario["network_type"],
        n_nodes=scenario["n_nodes"],
        p=scenario["p"],
        m=scenario["m"],
        batch_size=batch_size,
        init_noise_I=init_noise_I,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )

    return pd.DataFrame({"ratio": ratio_values, "X_mean": X_means})


def phase_sweep_df(
    max_workers: int | None = None,
    backend: str = "process",
    X0_values: Optional[np.ndarray] = None,
    ratio_values: Optional[np.ndarray] = None,
    scenario_kwargs: Optional[Dict] = None,
    batch_size: int = 16,
    init_noise_I: float = 0.04,
    T: int = 250,
    strategy_choice_func: str = "logit",
    tau: float = 1.0,
) -> pd.DataFrame:
    """Compute tidy DataFrame of X* over (X0, ratio)."""
    if X0_values is None:
        X0_values = np.linspace(0.0, 1.0, 21)
    if ratio_values is None:
        ratio_values = np.linspace(0.8, 3.5, 41)

    scenario = {
        "I0": 0.05,
        "beta_I": 2.0,
        "b": 1.0,
        "g_I": 0.05,
        "network_type": "BA",
        "n_nodes": 120,
        "p": 0.05,
        "m": 2,
    }
    if scenario_kwargs:
        scenario.update(scenario_kwargs)

    X_final = phase_sweep_X0_vs_ratio(
        X0_values,
        ratio_values,
        I0=scenario["I0"],
        beta_I=scenario["beta_I"],
        b=scenario["b"],
        g_I=scenario["g_I"],
        T=T,
        network_type=scenario["network_type"],
        n_nodes=scenario["n_nodes"],
        p=scenario["p"],
        m=scenario["m"],
        batch_size=batch_size,
        init_noise_I=init_noise_I,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
        max_workers=max_workers or 1,
        backend=backend,
    )

    rows = []
    for i, X0 in enumerate(X0_values):
        for j, ratio in enumerate(ratio_values):
            rows.append((float(X0), float(ratio), float(X_final[j, i])))
    return pd.DataFrame(rows, columns=["X0", "ratio", "X_final"])


def phase_sweep_X0_vs_I0_df(
    X0_values: Optional[np.ndarray] = None,
    I0_values: Optional[np.ndarray] = None,
    scenario_kwargs: Optional[Dict] = None,
    T: int = 150,
    n_seeds: int = 10,
    strategy_choice_func: str = "imitate",
    tau: float = 1.0,
    max_workers: int | None = None,
    backend: str = "process",
) -> pd.DataFrame:
    """Compute tidy DataFrame of mean final adoption X* over (X0, I0).

    Columns: ['X0', 'I0', 'X_final']
    """
    if X0_values is None:
        X0_values = np.linspace(0.0, 1.0, 15)
    if I0_values is None:
        I0_values = np.linspace(0.0, 1.0, 15)

    scenario = {
        "a0": 1.0,
        "beta_I": 2.5,
        "b": 1.0,
        "g_I": 0.1,
        "network_type": "random",
        "n_nodes": 200,
        "p": 0.03,
        "m": 3,
        "init_method": "random",
    }
    if scenario_kwargs:
        scenario.update(scenario_kwargs)

    X_final = phase_sweep_X0_vs_I0(
        X0_values=X0_values,
        I0_values=I0_values,
        a0=scenario["a0"],
        beta_I=scenario["beta_I"],
        b=scenario["b"],
        g_I=scenario["g_I"],
        T=T,
        network_type=scenario["network_type"],
        n_nodes=scenario["n_nodes"],
        p=scenario["p"],
        m=scenario["m"],
        n_seeds=n_seeds,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
        init_method=scenario["init_method"],
        max_workers=max_workers,
        backend=backend,
    )

    rows = []
    for i, I0 in enumerate(I0_values):
        for j, X0 in enumerate(X0_values):
            rows.append((float(X0), float(I0), float(X_final[i, j])))
    return pd.DataFrame(rows, columns=["X0", "I0", "X_final"])


def collect_topology_comparison_trials(
    X0_frac: float = 0.20,
    I0: float = 0.20,
    n_trials: int = 15,
    T: int = 150,
    n_nodes: int = 200,
    p: float = 0.03,
    m: int = 3,
    seed_base: int = 42,
    a0: float = 1.0,
    beta_I: float = 2.5,
    b: float = 1.0,
    g_I: float = 0.1,
    strategy_choice_func: str = "imitate",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Run ER vs BA comparison trials near the critical tipping threshold."""
    er_scenario = {
        "a0": a0,
        "beta_I": beta_I,
        "b": b,
        "g_I": g_I,
        "I0": I0,
        "network_type": "random",
        "n_nodes": n_nodes,
        "p": p,
        "X0_frac": X0_frac,
        "init_method": "random",
    }
    ba_scenario = {
        "a0": a0,
        "beta_I": beta_I,
        "b": b,
        "g_I": g_I,
        "I0": I0,
        "network_type": "BA",
        "n_nodes": n_nodes,
        "m": m,
        "X0_frac": X0_frac,
        "init_method": "random",
    }

    er_X, er_I, ba_X, ba_I = [], [], [], []
    for i in range(n_trials):
        seed = seed_base + i
        x_er, i_er, _ = run_timeseries_trial(T=T, scenario_kwargs=er_scenario, seed=seed, strategy_choice_func=strategy_choice_func)
        x_ba, i_ba, _ = run_timeseries_trial(T=T, scenario_kwargs=ba_scenario, seed=seed, strategy_choice_func=strategy_choice_func)
        er_X.append(x_er)
        er_I.append(i_er)
        ba_X.append(x_ba)
        ba_I.append(i_ba)

    def summarize(X_list: List[np.ndarray]) -> pd.DataFrame:
        mat = np.vstack(X_list)
        return pd.DataFrame({
            "X_mean": mat.mean(axis=0),
            "X_std": mat.std(axis=0),
            "X_med": np.median(mat, axis=0),
            "X_q10": np.quantile(mat, 0.10, axis=0),
            "X_q90": np.quantile(mat, 0.90, axis=0),
        })

    return er_X, er_I, ba_X, ba_I, summarize(er_X), summarize(ba_X)


def plot_intervention_fanchart(
    baseline_X: List[np.ndarray],
    subsidy_X: List[np.ndarray],
    out_path: Optional[str] = None,
) -> str:
    """Plot fan charts for baseline and subsidy trials and save to file.

    Returns the file path to the saved image.
    """
    T = len(baseline_X[0]) if baseline_X else 0
    t = np.arange(T)

    def quantiles(X_list: List[np.ndarray]):
        mat = np.vstack(X_list)
        return {
            "mean": mat.mean(axis=0),
            "q10": np.quantile(mat, 0.10, axis=0),
            "q25": np.quantile(mat, 0.25, axis=0),
            "q75": np.quantile(mat, 0.75, axis=0),
            "q90": np.quantile(mat, 0.90, axis=0),
            "final": mat[:, -1],
        }

    bq = quantiles(baseline_X)
    sq = quantiles(subsidy_X)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    # Baseline fan chart
    ax = axes[0, 0]
    ax.fill_between(t, bq["q10"], bq["q90"], color="steelblue", alpha=0.15, label="10–90%")
    ax.fill_between(t, bq["q25"], bq["q75"], color="steelblue", alpha=0.30, label="25–75%")
    for X in baseline_X:
        ax.plot(t, X, color="steelblue", alpha=0.10, linewidth=1)
    ax.plot(t, bq["mean"], color="steelblue", linewidth=2, label="mean")
    ax.set_title("Baseline adoption")
    ax.set_xlabel("Time")
    ax.set_ylabel("X(t)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")

    # Subsidy fan chart
    ax = axes[0, 1]
    ax.fill_between(t, sq["q10"], sq["q90"], color="darkorange", alpha=0.15, label="10–90%")
    ax.fill_between(t, sq["q25"], sq["q75"], color="darkorange", alpha=0.30, label="25–75%")
    for X in subsidy_X:
        ax.plot(t, X, color="darkorange", alpha=0.10, linewidth=1)
    ax.plot(t, sq["mean"], color="darkorange", linewidth=2, label="mean")
    ax.set_title("Subsidy adoption")
    ax.set_xlabel("Time")
    ax.set_ylabel("X(t)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")

    # Histograms of final X(T)
    axes[1, 0].hist(bq["final"], bins=20, color="steelblue", alpha=0.8)
    axes[1, 0].set_title("Baseline final adoption X(T)")
    axes[1, 0].set_xlabel("X(T)")
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].hist(sq["final"], bins=20, color="darkorange", alpha=0.8)
    axes[1, 1].set_title("Subsidy final adoption X(T)")
    axes[1, 1].set_xlabel("X(T)")
    axes[1, 1].set_ylabel("Count")

    # Save figure
    if out_path is None:
        out_path = os.path.join(os.getcwd(), "ev_intervention_fanchart.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_spaghetti_traces(
    baseline_X: List[np.ndarray],
    subsidy_X: List[np.ndarray],
    *,
    max_traces: int = 100,
    alpha: float = 0.15,
    out_path: Optional[str] = None,
) -> str:
    """Plot raw trajectories as thin, transparent lines for baseline/subsidy.

    Shows bifurcation visually: many lines diverging toward 0 or 1 over time.
    """
    # Select random subset for visual clarity
    rng = np.random.default_rng(123)
    def subset(trajs: List[np.ndarray]) -> List[np.ndarray]:
        if len(trajs) <= max_traces:
            return trajs
        idx = rng.choice(len(trajs), size=max_traces, replace=False)
        return [trajs[i] for i in idx]

    b_sub = subset(baseline_X)
    s_sub = subset(subsidy_X)

    T = len(b_sub[0]) if b_sub else 0
    t = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    ax = axes[0]
    for X in b_sub:
        ax.plot(t, X, color="steelblue", alpha=alpha, linewidth=0.8)
    ax.set_title("Baseline traces")
    ax.set_xlabel("Time")
    ax.set_ylabel("X(t)")
    ax.set_ylim(0, 1)

    ax = axes[1]
    for X in s_sub:
        ax.plot(t, X, color="darkorange", alpha=alpha, linewidth=0.8)
    ax.set_title("Subsidy traces")
    ax.set_xlabel("Time")
    ax.set_ylabel("X(t)")
    ax.set_ylim(0, 1)

    if out_path is None:
        out_path = os.path.join(os.getcwd(), "ev_spaghetti.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_time_evolving_density(
    baseline_X: List[np.ndarray],
    subsidy_X: List[np.ndarray],
    *,
    x_bins: int = 50,
    time_bins: Optional[int] = None,
    out_path: Optional[str] = None,
) -> str:
    """Plot 2D histograms of (time, X) densities for baseline and subsidy.

    X-axis: time, Y-axis: adoption X(t), Color: frequency/density of passes.
    """
    if not baseline_X or not subsidy_X:
        raise ValueError("Need non-empty baseline and subsidy trajectories")

    T = len(baseline_X[0])
    if time_bins is None:
        time_bins = T

    # Flatten (t, X) points across all trials
    def flatten_points(trajs: List[np.ndarray]):
        t = np.arange(T)
        t_all = np.repeat(t, len(trajs))
        x_all = np.hstack(trajs)
        return t_all, x_all

    bt, bx = flatten_points(baseline_X)
    st, sx = flatten_points(subsidy_X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    hb = axes[0].hist2d(bt, bx, bins=[time_bins, x_bins], range=[[0, T - 1], [0.0, 1.0]], cmap="magma")
    axes[0].set_title("Baseline density: time vs X(t)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("X(t)")
    fig.colorbar(hb[3], ax=axes[0], label="count")

    hs = axes[1].hist2d(st, sx, bins=[time_bins, x_bins], range=[[0, T - 1], [0.0, 1.0]], cmap="magma")
    axes[1].set_title("Subsidy density: time vs X(t)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("X(t)")
    fig.colorbar(hs[3], ax=axes[1], label="count")

    if out_path is None:
        out_path = os.path.join(os.getcwd(), "ev_density.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def run_ratio_sweep_plot(
    X0_frac: float = 0.40,
    ratio_values: Optional[np.ndarray] = None,
    scenario_kwargs: Optional[Dict] = None,
    T: int = 250,
    batch_size: int = 16,
    init_noise_I: float = 0.04,
    strategy_choice_func: str = "logit",
    tau: float = 1.0,
    out_path: Optional[str] = None,
) -> str:
    """Sweep ratio values and plot final adoption X* vs a_I/b for a fixed X0.

    Calls the core computation helper and saves a simple line plot.
    Returns the path to the saved image.
    """
    scenario = {
        "beta_I": 2.0,
        "b": 1.0,
        "g_I": 0.05,
        "I0": 0.05,
        "network_type": "BA",
        "n_nodes": 120,
        "p": 0.05,
        "m": 2,
    }
    if scenario_kwargs:
        scenario.update(scenario_kwargs)

    if ratio_values is None:
        ratio_values = np.linspace(0.8, 3.5, 41)

    X_means = final_mean_adoption_vs_ratio(
        X0_frac,
        ratio_values,
        I0=scenario["I0"],
        beta_I=scenario["beta_I"],
        b=scenario["b"],
        g_I=scenario["g_I"],
        T=T,
        network_type=scenario["network_type"],
        n_nodes=scenario["n_nodes"],
        p=scenario["p"],
        m=scenario["m"],
        batch_size=batch_size,
        init_noise_I=init_noise_I,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ratio_values, X_means, color="tabblue" if hasattr(plt, "tabblue") else "C0", lw=2)
    ax.set_xlabel("a_I / b (ratio)")
    ax.set_ylabel("Final adoption X*")
    ax.set_title(f"X* vs ratio for X0={X0_frac:.2f}")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)

    if out_path is None:
        out_path = os.path.join(os.getcwd(), "ev_ratio_sweep.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_phase_plot_X0_vs_ratio_network(
    max_workers: int | None = None,
    backend: str = "process",
    X0_values: Optional[np.ndarray] = None,
    ratio_values: Optional[np.ndarray] = None,
    scenario_kwargs: Optional[Dict] = None,
    batch_size: int = 16,
    init_noise_I: float = 0.04,
    T: int = 250,
    strategy_choice_func: str = "logit",
    tau: float = 1.0,
    out_path: Optional[str] = None,
):
    """Produce a heatmap of X* over (X0, a_I/b) using core sweep helper.

    Saves a figure similar to the original model script and returns the path.
    """
    # Defaults aligned with the original phase plot
    if X0_values is None:
        X0_values = np.linspace(0.0, 1.0, 21)
    if ratio_values is None:
        ratio_values = np.linspace(0.8, 3.5, 41)

    scenario = {
        "I0": 0.05,
        "beta_I": 2.0,
        "b": 1.0,
        "g_I": 0.05,
        "network_type": "BA",
        "n_nodes": 120,
        "p": 0.05,
        "m": 2,
    }
    if scenario_kwargs:
        scenario.update(scenario_kwargs)

    X_final = phase_sweep_X0_vs_ratio(
        X0_values,
        ratio_values,
        I0=scenario["I0"],
        beta_I=scenario["beta_I"],
        b=scenario["b"],
        g_I=scenario["g_I"],
        T=T,
        network_type=scenario["network_type"],
        n_nodes=scenario["n_nodes"],
        p=scenario["p"],
        m=scenario["m"],
        batch_size=batch_size,
        init_noise_I=init_noise_I,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
        max_workers=max_workers or 1,
        backend=backend,
    )

    plt.figure(figsize=(7, 4))
    im = plt.imshow(
        X_final,
        origin="lower",
        extent=[X0_values[0], X0_values[-1], ratio_values[0], ratio_values[-1]],
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="plasma",
    )
    plt.colorbar(im, label="Final adopters X*")
    plt.xlabel("X0 (initial adoption)")
    plt.ylabel("a_I / b (initial payoff ratio)")
    plt.title("Network phase plot: X* over X0 and a_I/b")

    # Overlay initial threshold X = b/a_I => X = 1/ratio
    X_thresh = 1.0 / ratio_values
    X_thresh_clipped = np.clip(X_thresh, 0.0, 1.0)
    plt.plot(
        X_thresh_clipped,
        ratio_values,
        color="white",
        linestyle="--",
        linewidth=1.5,
        label="X = b / a_I (initial)",
    )

    plt.legend(loc="upper right")

    if out_path is None:
        out_path = os.path.join(os.getcwd(), "ev_phase_plot.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return out_path


def run_intervention_example(
    n_trials: int = 10,
    T: int = 200,
    scenario_kwargs: Optional[Dict] = None,
    subsidy_params: Optional[Dict] = None,
    max_workers: int = 1,
    seed_base: int = 42,
    strategy_choice_func: str = "imitate",
    tau: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Convenience: collect trials, plot, and return summary + image path."""

    baseline_X, baseline_I, subsidy_X, subsidy_I, baseline_df, subsidy_df = collect_intervention_trials(
        n_trials=n_trials,
        T=T,
        scenario_kwargs=scenario_kwargs,
        subsidy_params=subsidy_params,
        max_workers=max_workers,
        seed_base=seed_base,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )
    # Use DataFrame-based plotting to ensure outputs go to plots/
    traces_df = traces_to_long_df(baseline_X, subsidy_X)
    img_path = plot_fanchart(traces_df)
    return baseline_df, subsidy_df, img_path


# -----------------------------
# CLI Entrypoint
# -----------------------------

def main():
    # Defaults aligned with original ev_stag_mesa_model.run_intervention_example
    n_trials = 30  # use fewer than 500 for speed while keeping shape
    T = 200
    strategy_choice_func = "imitate"
    tau = 1.0
    max_workers = 1
    seed_base = 100

    scenario = dict(
        # Preserve initial ratio by computing a0 from ratio, matching the original
        ratio=2.3,
        beta_I=2.0,
        b=1.0,
        g_I=0.10,
        I0=0.05,
        network_type="BA",
        n_nodes=300,
        m=2,
        collect=True,
        X0_frac=0.40,
        init_method="random",
        # ER-specific `p` ignored for BA but kept for completeness
        p=0.05,
    )
    subsidy = dict(start=10, end=60, delta_a0=0.4, delta_beta_I=0.0)

    baseline_df, subsidy_df, img_path = run_intervention_example(
        n_trials=n_trials,
        T=T,
        scenario_kwargs=scenario,
        subsidy_params=subsidy,
        max_workers=max_workers,
        seed_base=seed_base,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )

    print("Baseline DF shape:", baseline_df.shape)
    print("Subsidy DF shape:", subsidy_df.shape)
    print("Saved image:", img_path)
    print("Baseline final X_mean:", float(baseline_df["X_mean"].iloc[-1]))
    print("Subsidy  final X_mean:", float(subsidy_df["X_mean"].iloc[-1]))

    # Also run the phase plot of X* over (X0, a_I/b) and save it
    phase_df = phase_sweep_df(
        max_workers=1,
        backend="thread",
        X0_values=np.linspace(0.0, 1.0, 21),
        ratio_values=np.linspace(0.8, 3.5, 31),
        batch_size=8,
        T=200,
        strategy_choice_func="logit",
        tau=1.0,
    )
    phase_path = plot_phase_plot(phase_df)
    print("Saved phase plot:", phase_path)

    # Spaghetti and time-evolving density plots
    # Use a larger trial count for clearer trace/density visuals
    n_trials_spaghetti = 100
    T_spaghetti = 200

    baseline_X, baseline_I, subsidy_X, subsidy_I, baseline_df2, subsidy_df2 = collect_intervention_trials(
        n_trials=n_trials_spaghetti,
        T=T_spaghetti,
        scenario_kwargs=scenario,
        subsidy_params=subsidy,
        max_workers=max_workers,
        seed_base=seed_base,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )
    traces_df = traces_to_long_df(baseline_X, subsidy_X)
    spaghetti_path = plot_spaghetti(traces_df, max_traces=100, alpha=0.15)
    print("Saved spaghetti plot:", spaghetti_path)

    density_path = plot_density(traces_df, x_bins=50, time_bins=T_spaghetti)
    print("Saved time-evolving density plot:", density_path)

    # Ratio sweep computed to DF then plotted
    sweep_df = ratio_sweep_df(
        X0_frac=scenario.get("X0_frac", 0.40),
        ratio_values=np.linspace(0.8, 3.5, 31),
        scenario_kwargs=scenario,
        T=200,
        batch_size=8,
        strategy_choice_func="logit",
        tau=1.0,
    )
    sweep_path = plot_ratio_sweep(sweep_df)
    print("Saved ratio sweep plot:", sweep_path)


if __name__ == "__main__":
    main()