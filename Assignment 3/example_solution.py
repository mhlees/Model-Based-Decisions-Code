"""
Assignment 3 Starter Script: EV Adoption, Coordination, and Policy Intervention
Course: Model-Based Decision Making (2026)
MSc Complex Systems & Policy, University of Amsterdam

This script provides the experimental scaffolding for the three tasks defined in `ev_stag_hunt.tex`:
  - Part 1: Baseline System Analysis (2D parameter sweep on a complete network to map the tipping boundary)
  - Part 2: Network Structure Analysis (ER vs. BA near the critical tipping boundary)
  - Part 3: Policy Intervention Design & Evaluation (Unforced Baseline vs. Two Competing Policies)

Instructions:
  - Several core sections are marked with `# TODO (Student): ...` where you must
    implement model parameter configurations, metric calculations, and policy interventions.
  - Run this script with Python (e.g., `python example_solution.py`) to generate Figures 1, 2, and 3.
  - Review `ev_core.py` and `ev_experiments.py` for underlying model rules and helper functions.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local module imports
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


# =====================================================================
# Part 1: Baseline System Analysis (Mean-Field Tipping Landscape)
# =====================================================================
def run_part1_baseline():
    """Part 1: Baseline System Analysis (Mean-Field Tipping Landscape).
    
    Tasks:
      1. Set up a 2D parameter grid over initial adoption X0 in [0, 1] and
         initial infrastructure I0 in [0, 1].
      2. Configure baseline model parameters on a fully connected network (K_200):
         N = 200, T = 150, a0 = 1.0, beta_I = 2.5, b = 1.0, g_I = 0.1, rule = "imitate".
      3. Execute a parameter sweep across an ensemble of stochastic seeds per cell.
      4. Save and plot Figure 1 (2D phase heatmap).
      5. Identify the critical tipping boundary X*(I0) and compare with the analytic prediction.
    """
    print("\n=======================================================")
    print("Part 1: Baseline System Analysis (2D Parameter Sweep)")
    print("=======================================================")
    
    # -----------------------------------------------------------------
    # TODO (Student): Define the 2D sweep grid
    # Choose a grid resolution (e.g., 15x15) spanning X0 in [0, 1] and I0 in [0, 1].
    # Hint: Use np.linspace(0.0, 1.0, n_grid).
    # -----------------------------------------------------------------
    n_grid = 15
    X0_values = np.linspace(0.0, 1.0, n_grid)  # TODO (Student): verify or adjust grid resolution
    I0_values = np.linspace(0.0, 1.0, n_grid)  # TODO (Student): verify or adjust grid resolution

    # -----------------------------------------------------------------
    # TODO (Student): Configure baseline parameters in scenario_kwargs
    # Set:
    #   - "a0": baseline coordination payoff (1.0)
    #   - "beta_I": infrastructure feedback sensitivity (2.5)
    #   - "b": defection payoff (1.0)
    #   - "g_I": infrastructure adjustment rate (0.1)
    #   - "network_type": fully connected complete network ("complete")
    #   - "n_nodes": population size (200)
    #   - "init_method": uniform random seeding ("random")
    # -----------------------------------------------------------------
    scenario_kwargs = {
        # TODO (Student): Fill in required model parameters
        "a0": 1.0,
        "beta_I": 2.5,
        "b": 1.0,
        "g_I": 0.1,
        "network_type": "complete",
        "n_nodes": 200,
        "init_method": "random",
    }
    
    T = 150
    n_seeds = 10  # Justify your chosen number of stochastic seeds in the report
    
    print(f"Sweeping a {n_grid}x{n_grid} grid over (X0, I0) on complete network with {n_seeds} seeds per cell...")
    
    # -----------------------------------------------------------------
    # TODO (Student): Execute the parameter sweep
    # Call phase_sweep_X0_vs_I0_df(...) with:
    #   - X0_values, I0_values
    #   - scenario_kwargs
    #   - T=T, n_seeds=n_seeds, strategy_choice_func="imitate"
    # -----------------------------------------------------------------
    phase_df = phase_sweep_X0_vs_I0_df(
        X0_values=X0_values,
        I0_values=I0_values,
        scenario_kwargs=scenario_kwargs,
        T=T,
        n_seeds=n_seeds,
        strategy_choice_func="imitate",
        max_workers=None,
        backend="process",
    )
    
    # Plot and save Figure 1
    out_path = os.path.join(script_dir, "plots", "figure1_phase_heatmap.png")
    plot_phase_X0_vs_I0(phase_df, out_path=out_path)
    print(f"[Figure 1 Generated]: {out_path}")
    
    # -----------------------------------------------------------------
    # TODO (Student): Inspect the tipping boundary & compare with theory
    # Pivot phase_df into a 2D matrix (index=I0, columns=X0, values=X_final)
    # and compare the empirical tipping boundary against the theoretical formula:
    #   X^*(I) = b / (a0 + beta_I * I)
    # -----------------------------------------------------------------
    pivot = phase_df.pivot(index="I0", columns="X0", values="X_final")
    print("\nSummary of Mean Final Adoption X*(X0, I0) [corner sample]:")
    print(pivot.iloc[::3, ::3].round(2))
    
    print("\n--- Theoretical Mean-Field Tipping Boundary X*(I) = b / (a0 + beta_I * I) ---")
    for I_val in [0.0, 0.25, 0.50, 0.75, 1.0]:
        a_I = 1.0 + 2.5 * I_val
        X_star = 1.0 / a_I
        print(f"  At I = {I_val:.2f}: a(I) = {a_I:.3f} -> Analytic Tipping Threshold X* = {X_star:.3f}")
    
    return phase_df


# =====================================================================
# Part 2: Network Structure Analysis (Random vs. Scale-Free)
# =====================================================================
def run_part2_network_comparison(phase_df=None):
    """Part 2: Network Structure Analysis (Random vs. Scale-Free).
    
    Tasks:
      1. Identify a critical initial condition (X0_crit, I0_crit) located near
         the tipping boundary found in Part 1.
      2. Configure two contrasting networks of identical size (N=200) and
         comparable average degree (<k> ~ 6):
           - Erdős–Rényi (p ~ 0.03)
           - Barabási–Albert (m = 3)
      3. Run an ensemble of stochastic replications (e.g., 20 trials) for each topology.
      4. Generate Figure 2 (comparative dynamic time-series traces of X(t) and I(t)).
      5. Compute quantitative comparison metrics:
           - Success rate: P(final adoption > 0.5)
           - Settling time: time step where adoption reaches steady state
    """
    print("\n=======================================================")
    print("Part 2: Network Structure Analysis (ER vs. BA near Tipping)")
    print("=======================================================")
    
    # -----------------------------------------------------------------
    # TODO (Student): Select a critical initial condition (X0_crit, I0_crit)
    # Choose a point near the tipping boundary identified in Part 1.
    # In this regime, stochastic fluctuations determine whether the system
    # tips to full adoption (X* ~ 1) or collapses to defection (X* ~ 0).
    # -----------------------------------------------------------------
    X0_crit = 0.28  # TODO (Student): Select based on Part 1 heatmap findings
    I0_crit = 0.30  # TODO (Student): Select based on Part 1 heatmap findings
    
    n_trials = 20   # Ensemble size to capture stochastic divergence
    T = 150
    
    print(f"Running {n_trials} trials near tipping boundary (X0={X0_crit}, I0={I0_crit})...")
    
    # -----------------------------------------------------------------
    # TODO (Student): Configure and collect trials for ER and BA networks
    # Verify the parameters that control average degree:
    #   - For ER: p = 0.03 gives <k> = (N-1)*p = 199 * 0.03 ~ 6
    #   - For BA: m = 3 gives <k> ~ 2*m = 6
    # Call collect_topology_comparison_trials(...)
    # -----------------------------------------------------------------
    er_X, er_I, ba_X, ba_I, er_summary, ba_summary = collect_topology_comparison_trials(
        X0_frac=X0_crit,
        I0=I0_crit,
        n_trials=n_trials,
        T=T,
        n_nodes=200,
        p=0.03,  # TODO (Student): Confirm ER edge probability for <k> ~ 6
        m=3,     # TODO (Student): Confirm BA m parameter for <k> ~ 6
        seed_base=100,
        a0=1.0,
        beta_I=2.5,
        b=1.0,
        g_I=0.1,
        strategy_choice_func="imitate",
    )
    
    # Generate Figure 2
    out_path = os.path.join(script_dir, "plots", "figure2_topology_comparison.png")
    plot_topology_comparison(er_X, er_I, ba_X, ba_I, out_path=out_path)
    print(f"[Figure 2 Generated]: {out_path}")
    
    # -----------------------------------------------------------------
    # TODO (Student): Compute quantitative comparison metrics
    # Calculate:
    #   1. Transition success probability: fraction of runs where final adoption > 0.5.
    #   2. Settling time: time step t* at which the trajectory settles to steady state.
    #      (e.g., first time t where |X(t') - X_final| < 0.02 for all t' >= t).
    # -----------------------------------------------------------------
    er_finals = np.array([x[-1] for x in er_X])
    ba_finals = np.array([x[-1] for x in ba_X])
    
    er_success = np.mean(er_finals > 0.5) * 100
    ba_success = np.mean(ba_finals > 0.5) * 100
    
    def compute_settling_time(x_series, threshold=0.02, window=10):
        """Compute the time step where trajectory settles within threshold of final value."""
        final_val = x_series[-1]
        for t in range(len(x_series) - window):
            if np.all(np.abs(x_series[t:] - final_val) < threshold):
                return t
        return len(x_series)

    er_settle_times = [compute_settling_time(x) for x in er_X]
    ba_settle_times = [compute_settling_time(x) for x in ba_X]
    
    print("\n--- Quantitative Topological Comparison ---")
    print(f"ER  Transition Success Rate: {er_success:.1f}% | Median Settling Time: {np.median(er_settle_times):.1f} steps")
    print(f"BA  Transition Success Rate: {ba_success:.1f}% | Median Settling Time: {np.median(ba_settle_times):.1f} steps")
    
    return er_X, er_I, ba_X, ba_I


# =====================================================================
# Part 3: Policy Intervention Design and Evaluation
# =====================================================================
def run_part3_policy_evaluation():
    """Part 3: Policy Intervention Design and Evaluation.
    
    Tasks:
      1. Set initial conditions in a sub-critical regime (just below tipping),
         where the unforced baseline locks into low adoption (X* ~ 0).
      2. Choose ONE network topology (e.g., Barabási–Albert) and justify it.
      3. Design TWO competing policy interventions:
           - Policy 1 (e.g., Early Temporary Payoff Subsidy or Capital Shock)
           - Policy 2 (e.g., Late Policy or alternate intensity)
      4. Simulate an ensemble of stochastic seeds for:
           - Unforced Baseline
           - Policy 1
           - Policy 2
      5. Generate Figure 3 (comparative evaluation) and evaluate trade-offs
         between timing, duration, intensity, and fiscal cost.
    """
    print("\n=======================================================")
    print("Part 3: Policy Intervention Design and Evaluation")
    print("=======================================================")
    
    # Sub-critical initial conditions (system fails without policy intervention)
    X0_subcrit = 0.18
    I0_subcrit = 0.20
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
    
    print(f"Simulating {n_trials} seeds on sub-critical BA network (X0={X0_subcrit}, I0={I0_subcrit})...")
    
    # -----------------------------------------------------------------
    # 1. Unforced Baseline (Provided as reference)
    # -----------------------------------------------------------------
    print("Running Unforced Baseline...")
    baseline_X = []
    for i in range(n_trials):
        seed = seed_base + i
        x, _, _ = run_timeseries_trial(
            T=T,
            scenario_kwargs=scenario,
            seed=seed,
            strategy_choice_func="imitate",
        )
        baseline_X.append(x)
        
    # -----------------------------------------------------------------
    # TODO (Student): Design and evaluate Policy Intervention 1
    # Choose a policy lever from:
    #   - Temporary Payoff Subsidy: policy_subsidy_factory(start, end, delta_a0)
    #   - Infrastructure Capital Shock: policy_infrastructure_boost_factory(shock_time, delta_I)
    #
    # Example (uncomment and configure):
    #   policy_1 = policy_subsidy_factory(start=0, end=30, delta_a0=0.60)
    # -----------------------------------------------------------------
    policy_1 = None  # TODO (Student): Define Policy 1
    
    policy_1_X = []
    if policy_1 is not None:
        print("Running Policy 1 Trials...")
        for i in range(n_trials):
            seed = seed_base + i
            # TODO (Student): Execute run_timeseries_trial with policy=policy_1
            x, _, _ = run_timeseries_trial(
                T=T,
                scenario_kwargs=scenario,
                seed=seed,
                policy=policy_1,
                strategy_choice_func="imitate",
            )
            policy_1_X.append(x)
    else:
        print("\n[TODO (Student)]: Policy 1 is not defined yet.")
        print("  Formulate policy_1 (e.g., using policy_subsidy_factory or policy_infrastructure_boost_factory).")
        
    # -----------------------------------------------------------------
    # TODO (Student): Design and evaluate Policy Intervention 2
    # Design a competing policy intervention to compare against Policy 1.
    # For example:
    #   - Test Timing: Late Subsidy (e.g., active during steps [30, 60)) vs Early Subsidy [0, 30)
    #   - OR Test Intensity: Higher/Lower subsidy, or Infrastructure Capital Shock
    #
    # Example (uncomment and configure):
    #   policy_2 = policy_subsidy_factory(start=30, end=60, delta_a0=0.60)
    # -----------------------------------------------------------------
    policy_2 = None  # TODO (Student): Define Policy 2
    
    policy_2_X = []
    if policy_2 is not None:
        print("Running Policy 2 Trials...")
        for i in range(n_trials):
            seed = seed_base + i
            # TODO (Student): Execute run_timeseries_trial with policy=policy_2
            x, _, _ = run_timeseries_trial(
                T=T,
                scenario_kwargs=scenario,
                seed=seed,
                policy=policy_2,
                strategy_choice_func="imitate",
            )
            policy_2_X.append(x)
    else:
        print("[TODO (Student)]: Policy 2 is not defined yet.")
        print("  Formulate a competing policy_2 to evaluate timing or intensity trade-offs.")

    # -----------------------------------------------------------------
    # TODO (Student): Generate Figure 3 and compute policy trade-off metrics
    # Compare:
    #   - Final adoption outcome X(T)
    #   - Policy success probability: P(X(T) > 0.5)
    #   - Policy cost / duration efficiency
    # -----------------------------------------------------------------
    if policy_1_X:
        out_path = os.path.join(script_dir, "plots", "figure3_policy_evaluation.png")
        plot_intervention_comparison(
            baseline_X=baseline_X,
            intervention_X=policy_1_X,
            policy_label=r"Policy 1 Intervention",
            timing_window=(0, 30),
            out_path=out_path,
            title="Figure 3: Policy Intervention vs. Unforced Baseline (BA Network)",
        )
        print(f"[Figure 3 Generated]: {out_path}")
        
        b_finals = np.array([x[-1] for x in baseline_X])
        p1_finals = np.array([x[-1] for x in policy_1_X])
        print("\n--- Policy Performance Metrics ---")
        print(f"Unforced Baseline: Mean Final X* = {np.mean(b_finals):.3f} (Success: {np.mean(b_finals > 0.5)*100:.0f}%)")
        print(f"Policy 1:          Mean Final X* = {np.mean(p1_finals):.3f} (Success: {np.mean(p1_finals > 0.5)*100:.0f}%)")
        if policy_2_X:
            p2_finals = np.array([x[-1] for x in policy_2_X])
            print(f"Policy 2:          Mean Final X* = {np.mean(p2_finals):.3f} (Success: {np.mean(p2_finals > 0.5)*100:.0f}%)")
    else:
        print("\n[Notice]: Define and run Policy 1 to generate Figure 3 and print policy metrics.")


# =====================================================================
# Main Execution Entry Point
# =====================================================================
def main():
    os.makedirs(os.path.join(script_dir, "plots"), exist_ok=True)
    print("=================================================================")
    print("Model-Based Decision Making (2026) — Assignment 3 Starter Script")
    print("=================================================================")
    
    # Execute the three assignment parts
    phase_df = run_part1_baseline()
    run_part2_network_comparison(phase_df)
    run_part3_policy_evaluation()
    
    print("\nAll 3 figures successfully generated and saved to Code/Assignment 3/plots/")


if __name__ == "__main__":
    main()
