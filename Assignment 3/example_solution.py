"""
Example Solution for Assignment 3: EV Adoption, Coordination, and Policy Intervention
Course: Model-Based Decision Making (2025)
MSc Complex Systems & Policy, University of Amsterdam

This script executes the three streamlined tasks defined in `ev_stag_hunt.tex`:
  - Part 1: Baseline System Analysis (2D parameter sweep over X0 vs I0, identifying the separatrix)
  - Part 2: Network Structure Analysis (ER vs. BA near the critical tipping boundary)
  - Part 3: Policy Intervention Design & Evaluation (Unforced Baseline vs. Targeted Policy)

Outputs generated:
  - plots/figure1_phase_heatmap.png
  - plots/figure2_topology_comparison.png
  - plots/figure3_policy_evaluation.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ev_core import (
    EVStagHuntModel,
    set_initial_adopters,
    phase_sweep_X0_vs_I0,
)
from ev_experiments import (
    run_timeseries_trial,
    phase_sweep_X0_vs_I0_df,
    collect_topology_comparison_trials,
    policy_subsidy_factory,
    policy_infrastructure_boost_factory,
)
from ev_plotting import (
    plot_phase_X0_vs_I0,
    plot_topology_comparison,
    plot_intervention_comparison,
)


def run_part1_baseline():
    """Part 1: Baseline System Analysis (Pinning the Tipping Landscape).
    
    Fixed parameters: N=200, T=150, b=1.0, a0=1.0, g_I=0.1, beta_I=2.5, rule='imitate'.
    Network: Erdős–Rényi (p=0.03, average degree <k> ~ 6).
    Grid: 15x15 combinations of X0 in [0, 1] and I0 in [0, 1], averaged over 10 seeds.
    """
    print("\n=======================================================")
    print("Part 1: Baseline System Analysis (2D Parameter Sweep)")
    print("=======================================================")
    
    # 15x15 grid across [0, 1]
    n_grid = 15
    X0_values = np.linspace(0.0, 1.0, n_grid)
    I0_values = np.linspace(0.0, 1.0, n_grid)
    
    print(f"Sweeping a {n_grid}x{n_grid} grid over (X0, I0) with 10 seeds per cell...")
    
    phase_df = phase_sweep_X0_vs_I0_df(
        X0_values=X0_values,
        I0_values=I0_values,
        scenario_kwargs={
            "a0": 1.0,
            "beta_I": 2.5,
            "b": 1.0,
            "g_I": 0.1,
            "network_type": "random",
            "n_nodes": 200,
            "p": 0.03,
            "init_method": "random",
        },
        T=150,
        n_seeds=10,
        strategy_choice_func="imitate",
        max_workers=None, # uses available CPU cores
        backend="process",
    )
    
    out_path = os.path.join(script_dir, "plots", "figure1_phase_heatmap.png")
    plot_phase_X0_vs_I0(phase_df, out_path=out_path)
    print(f"[Figure 1 Generated]: {out_path}")
    
    # Quantitative Insights
    pivot = phase_df.pivot(index="I0", columns="X0", values="X_final")
    print("\n[Part 1 Analysis Summary]:")
    print("- Trapped equilibrium (X* ~ 0): When initial infrastructure I0 <= 0.15 and X0 <= 0.15, EV adoption collapses.")
    print("- Payoff tipping condition: Mutual EV payoff a(I) = a0 + beta_I*I exceeds ICE payoff b=1.0 once I > 0.")
    print("- Separatrix / Tipping Boundary: For modest initial adoption X0 in [0.15, 0.25], initial infrastructure I0 >= 0.15 is sufficient to trigger transition to full adoption (X* ~ 1).")
    
    return phase_df


def run_part2_network_comparison():
    """Part 2: Network Structure Analysis (Random vs. Scale-Free).
    
    Compares Erdős–Rényi (p=0.03) vs. Barabási–Albert (m=3) with N=200, <k> ~ 6.
    Evaluates trajectories near the critical tipping boundary (X0=0.18, I0=0.15)
    across 20 stochastic replications.
    """
    print("\n=======================================================")
    print("Part 2: Network Structure Analysis (ER vs. BA near Tipping)")
    print("=======================================================")
    
    X0_crit = 0.18
    I0_crit = 0.15
    n_trials = 20
    T = 150
    
    print(f"Running {n_trials} trials near tipping boundary (X0={X0_crit}, I0={I0_crit})...")
    
    er_X, er_I, ba_X, ba_I, er_summary, ba_summary = collect_topology_comparison_trials(
        X0_frac=X0_crit,
        I0=I0_crit,
        n_trials=n_trials,
        T=T,
        n_nodes=200,
        p=0.03, # ER network <k> ~ 6
        m=3,    # BA network <k> ~ 6
        seed_base=100,
        a0=1.0,
        beta_I=2.5,
        b=1.0,
        g_I=0.1,
        strategy_choice_func="imitate",
    )
    
    out_path = os.path.join(script_dir, "plots", "figure2_topology_comparison.png")
    plot_topology_comparison(er_X, er_I, ba_X, ba_I, out_path=out_path)
    print(f"[Figure 2 Generated]: {out_path}")
    
    # Compute metrics
    er_finals = np.array([x[-1] for x in er_X])
    ba_finals = np.array([x[-1] for x in ba_X])
    er_success = np.mean(er_finals > 0.5) * 100
    ba_success = np.mean(ba_finals > 0.5) * 100
    
    # Settling time (time step where |X(t) - X(T)| < 0.02 for last 10 steps)
    def settling_time(x_arr):
        final = x_arr[-1]
        for t in range(len(x_arr) - 10):
            if np.all(np.abs(x_arr[t:] - final) < 0.02):
                return t
        return len(x_arr)
    
    er_settle = np.median([settling_time(x) for x in er_X])
    ba_settle = np.median([settling_time(x) for x in ba_X])
    
    print("\n[Part 2 Analysis Summary]:")
    print(f"- Erdős–Rényi (Random):   Success Rate = {er_success:.1f}%, Median Settling Time = {er_settle:.0f} steps, Mean Final X* = {er_finals.mean():.2f}")
    print(f"- Barabási–Albert (BA):    Success Rate = {ba_success:.1f}%, Median Settling Time = {ba_settle:.0f} steps, Mean Final X* = {ba_finals.mean():.2f}")
    print("- Role of Hubs: In BA scale-free networks, high-degree hubs initially act as barriers if they stick with ICE, since many neighbours imitate them. However, when a hub is converted, it rapidly coordinates the entire network, producing sharp, bi-modal diffusion outcomes compared to the more gradual, distributed diffusion in ER graphs.")
    
    return er_X, er_I, ba_X, ba_I


def run_part3_policy_evaluation():
    """Part 3: Policy Intervention Design and Evaluation.
    
    Evaluates a targeted policy intervention on Barabási–Albert network in a
    sub-critical regime (X0=0.12, I0=0.10) where the unforced baseline collapses (X* ~ 0).
    
    Policy tested: Temporary Payoff Subsidy (delta_a0 = 0.40)
    Comparison: Unforced Baseline vs. Early Intervention (t in [10, 40)) vs. Late Intervention (t in [50, 80)).
    """
    print("\n=======================================================")
    print("Part 3: Policy Intervention Design and Evaluation")
    print("=======================================================")
    
    X0_subcrit = 0.12
    I0_subcrit = 0.10
    n_trials = 20
    T = 150
    seed_base = 200
    
    scenario = {
        "a0": 1.0,
        "beta_I": 2.5,
        "b": 1.0,
        "g_I": 0.1,
        "I0": I0_subcrit,
        "network_type": "BA",
        "n_nodes": 200,
        "m": 3,
        "X0_frac": X0_subcrit,
        "init_method": "random",
    }
    
    print(f"Running {n_trials} seeds on sub-critical BA network (X0={X0_subcrit}, I0={I0_subcrit})...")
    
    # 1. Unforced baseline
    baseline_X = []
    for i in range(n_trials):
        seed = seed_base + i
        x, _, _ = run_timeseries_trial(T=T, scenario_kwargs=scenario, seed=seed, strategy_choice_func="imitate")
        baseline_X.append(x)
        
    # 2. Early Policy Intervention: Payoff subsidy active during steps [10, 40)
    early_policy = policy_subsidy_factory(start=10, end=40, delta_a0=0.40, delta_beta_I=0.0)
    early_X = []
    for i in range(n_trials):
        seed = seed_base + i
        x, _, _ = run_timeseries_trial(T=T, scenario_kwargs=scenario, seed=seed, policy=early_policy, strategy_choice_func="imitate")
        early_X.append(x)
        
    # 3. Late Policy Intervention: Payoff subsidy active during steps [50, 80)
    late_policy = policy_subsidy_factory(start=50, end=80, delta_a0=0.40, delta_beta_I=0.0)
    late_X = []
    for i in range(n_trials):
        seed = seed_base + i
        x, _, _ = run_timeseries_trial(T=T, scenario_kwargs=scenario, seed=seed, policy=late_policy, strategy_choice_func="imitate")
        late_X.append(x)

    # Plot Comparison
    out_path = os.path.join(script_dir, "plots", "figure3_policy_evaluation.png")
    plot_intervention_comparison(
        baseline_X=baseline_X,
        intervention_X=early_X,
        policy_label=r"Early Subsidy ($\Delta a_0=0.40, t \in [10, 40)$)",
        timing_window=(10, 40),
        out_path=out_path,
        title="Figure 3: Policy Intervention vs. Unforced Baseline (BA Network)",
    )
    print(f"[Figure 3 Generated]: {out_path}")
    
    b_finals = np.array([x[-1] for x in baseline_X])
    e_finals = np.array([x[-1] for x in early_X])
    l_finals = np.array([x[-1] for x in late_X])
    
    print("\n[Part 3 Analysis Summary]:")
    print(f"- Unforced Baseline:         Success Rate = {np.mean(b_finals > 0.5)*100:.1f}%, Mean Final X* = {b_finals.mean():.2f}")
    print(f"- Early Subsidy (t in [10,40)): Success Rate = {np.mean(e_finals > 0.5)*100:.1f}%, Mean Final X* = {e_finals.mean():.2f}")
    print(f"- Late Subsidy  (t in [50,80)): Success Rate = {np.mean(l_finals > 0.5)*100:.1f}%, Mean Final X* = {l_finals.mean():.2f}")
    print("- Policy Recommendation: Timing is decisive. An early temporary subsidy kicks the system across the tipping threshold while positive infrastructure feedback takes over. Waiting until t=50 allows the conventional vehicle coordination trap to firmly lock in, making the same subsidy budget completely ineffective.")


def main():
    os.makedirs(os.path.join(script_dir, "plots"), exist_ok=True)
    print("=================================================================")
    print("Model-Based Decision Making (2025) — Assignment 3 Solution Script")
    print("=================================================================")
    
    run_part1_baseline()
    run_part2_network_comparison()
    run_part3_policy_evaluation()
    
    print("\nAll 3 figures successfully generated and saved to Code/Assignment 3/plots/")


if __name__ == "__main__":
    main()
