# EV Stag Hunt Model — Assignment 3

This folder contains a network-based agent model of EV adoption framed as a Stag Hunt coordination game, plus experiment runners and plotting utilities. It is designed for teaching and exploration of how infrastructure, incentives, network structure, and decision rules shape adoption dynamics.

## What’s Included
- `ev_core.py` — Core Mesa model and helpers: agents, network generation, step dynamics, initial adopters, and computation utilities for ratio/phase sweeps.
- `ev_experiments.py` — Experiment orchestrators: trial runners, policy factories, data-frame builders, CLI `main()` that generates example plots.
- `ev_plotting.py` — Plotting utilities that take tidy pandas DataFrames and save figures to `plots/`.
- `plots/` — Output directory for generated figures.

## Quick Start
- Install dependencies from the repo root:
  - `pip install -r Code/requirements.txt`
- Run the example experiments from this folder:
  - `cd Code/Assignment\ 3`
  - `python ev_experiments.py`
- Outputs:
  - Plots are saved under `Code/Assignment 3/plots/` by default (see “Outputs and Plots”).
  - Some legacy plots from `ev_experiments.py` may save to the current directory; prefer `ev_plotting.py` for consistent `plots/` outputs.

## Model Overview (ev_core.py)
- Agents and Strategies
  - Each node hosts one agent with strategy `"C"` (adopt EV) or `"D"` (stick with ICE).
  - Agents interact with neighbors to accumulate payoffs, then update strategies synchronously.
- Payoffs (Degree-Normalised)
  - Coordination payoff for `C` when meeting another `C`: `a_I = a0 + beta_I * I`.
  - Meeting `D`: the `C` agent gets `0`, the `D` agent gets `b`.
  - Two `D` agents each get `b`.
  - **Degree Normalisation:** To enable fair comparison across topologies (from complete graphs with $k=199$ to sparse networks with $\langle k \rangle \approx 6$) and mitigate hub degree bias, each agent's round payoff is divided by its node degree $k_i$:
    - For defection: $\bar{\pi}_i(D) = b = 1.0$ (constant).
    - For adoption: $\bar{\pi}_i(C) = x_i \cdot a(I)$, where $x_i = k_{i,C} / k_i$ is the local fraction of EV adopters among neighbours.
- Infrastructure Dynamics
  - Infrastructure level `I` is a global scalar in `[0, 1]`.
  - Update rule after all agents advance: `I(t+1) = clip(I(t) + g_I * (X(t) - I(t)), 0, 1)`, where `X` is the fraction of `C` agents.
  - Intuition: if adoption `X` exceeds infrastructure `I`, infrastructure grows toward `X` at rate `g_I`; otherwise it relaxes downward.
- Strategy Choice Rules
  - `imitate`: pick the strategy of the highest-payoff neighbor (including self).
  - `logit`: probabilistic choice via softmax with temperature `tau`:
    - Compute expected payoffs vs neighbors, then choose `C` with probability `exp(pi_C/tau) / (exp(pi_C/tau) + exp(pi_D/tau))`.
- Network Structures
  - `complete` (Fully connected): all-to-all connectivity (mean-field baseline for Part 1).
  - `random` (Erdős–Rényi): parameters `n_nodes`, edge probability `p`.
  - `BA` (Barabási–Albert): parameters `n_nodes`, `m` edges attached per new node.
- Data Collection
  - Model-level: `X` (adoption fraction), `I` (infrastructure level) per time step.
  - Agent-level: `strategy`, `payoff` per time step when `collect=True`.

## Key Parameters
- `a0` — Base coordination payoff for EV adoption.
- `beta_I` — Infrastructure sensitivity of coordination payoff.
- `b` — Payoff from ICE (defection).
- `g_I` — Infrastructure adjustment rate toward current adoption.
- `I0` — Initial infrastructure level.
- `n_nodes` — Number of agents in the network.
- `network_type` — `"complete"`, `"random"`, or `"BA"`.
- `p` — Edge probability (ER only).
- `m` — Edges attached per new node (BA only).
- `strategy_choice_func` — `"imitate"` or `"logit"`.
- `tau` — Temperature for the logit rule (higher = more exploration/randomness).
- `collect` — Enable Mesa `DataCollector` for model and agent variables.

## Initial Adoption (X0)
- Use `set_initial_adopters(model, X0_frac, method, seed, high)` to seed EV adopters:
  - `method="random"` — uniformly random agents become `C`.
  - `method="degree"` — select by node degree; `high=True` chooses hubs, `high=False` chooses low-degree nodes.
  - feel free to add more methods!
- Alternatively, you can create the model with `initial_ev=int(round(X0_frac * n_nodes))` for quick trials (used in sweeps).

## Experiment Runners (ev_experiments.py)
- Single Trial
  - `run_timeseries_trial(T, scenario_kwargs, seed, policy, strategy_choice_func, tau)`
  - Returns `(X_series, I_series, df)` where `df` is the model DataFrame.
  - `scenario_kwargs` can include network, payoff, and initialization fields; if `ratio` is provided, the code sets `a0 = ratio * b - beta_I * I0` to preserve the intended initial payoff ratio.
- Parallel Trials: Baseline vs Subsidy
  - `collect_intervention_trials(n_trials, T, scenario_kwargs, subsidy_params, max_workers, seed_base, strategy_choice_func, tau)`
  - Runs `n_trials` with and without a policy and returns raw trajectories plus summary DataFrames.
  - Policy spec passed as a dict (`{"type": "subsidy" | "infrastructure", "params": {...}}`) is reconstructed inside worker processes.
- DataFrame Builders
  - `traces_to_long_df(baseline_X, subsidy_X)` → tidy `[group, trial, time, X]` for spaghetti/density plots.
  - `ratio_sweep_df(X0_frac, ratio_values, scenario_kwargs, T, batch_size, strategy_choice_func, tau)` → `[ratio, X_mean]` for X* vs ratio.
  - `phase_sweep_df(max_workers, backend, X0_values, ratio_values, scenario_kwargs, batch_size, init_noise_I, T, strategy_choice_func, tau)` → `[X0, ratio, X_final]` for phase heatmaps.
- Convenience
  - `run_intervention_example(...)` runs baseline vs subsidy, produces a fan chart, and returns summary frames and the image path.
- CLI Entrypoint
  - `main()` demonstrates phase plot, spaghetti, time-evolving density, and ratio sweep using the DataFrame builders and plotting utilities.

## Policy Interventions
- Included Policy Factories
  - `policy_subsidy_factory(start, end, delta_a0=0.3, delta_beta_I=0.0)` — temporarily increase coordination payoff (`a0`) and/or infrastructure sensitivity (`beta_I`) between `start` and `end`.
  - `policy_infrastructure_boost_factory(start, boost=0.2, once=True)` — inject a one-off or repeated infrastructure increase.
- How to Use Policies
  - Directly with a single trial:
    - ```python
      policy = policy_subsidy_factory(start=10, end=60, delta_a0=0.4)
      X, I, df = run_timeseries_trial(T=200, scenario_kwargs=scenario, policy=policy)
      ```
  - In parallel trial collection, pass a spec dict:
    - ```python
      subsidy = {"type": "subsidy", "params": {"start": 10, "end": 60, "delta_a0": 0.4}}
      baseline_X, baseline_I, subsidy_X, subsidy_I, base_df, subs_df = collect_intervention_trials(
          n_trials=100, T=200, scenario_kwargs=scenario, subsidy_params={"start": 10, "end": 60, "delta_a0": 0.4})
      ```
- Additional Intervention Ideas
  - Change `g_I` to represent faster/slower infrastructure response.
  - Adjust `b` to model fuel taxes or changing ICE appeal.
  - Information campaigns via `strategy_choice_func="logit"` and higher `tau` (more exploration).
  - Targeted seeding: use `set_initial_adopters(..., method="degree", high=True)` to seed hubs.
  - Time-windowed policies chaining subsidy and infrastructure boosts.

## Networks You Can Use
- Random (ER): set `network_type="random"`, choose `n_nodes` and `p`.
- Scale-free (BA): set `network_type="BA"`, choose `n_nodes` and `m`.
- Comparative experiments: run the same scenario under both to study robustness.

## Outputs and Plots (ev_plotting.py)
All plotting functions accept tidy DataFrames and save figures. By default, outputs go to `plots/`.

- Spaghetti (Trace) Plot — `plot_spaghetti(traces_df, max_traces=..., alpha=...)`
  - Input: columns `[group, trial, time, X]` from `traces_to_long_df`.
  - Shows many raw trajectories with high transparency to reveal bifurcations.
  - Use `max_traces` and `alpha` to tune clarity.
- Time-Evolving Density (2D Histogram/Heatmap) — `plot_density(traces_df, x_bins=..., time_bins=...)`
  - Input: same as spaghetti; bins over time and adoption level `X`.
  - Shows where trajectories concentrate over time (probability mass of adoption).
  - Increase `time_bins` to match the horizon; adjust `x_bins` for resolution.
- Ratio Sweep — `plot_ratio_sweep(sweep_df)`
  - Input: `[ratio, X_mean]` from `ratio_sweep_df`.
  - Plots final adoption `X*` vs initial payoff ratio `a_I/b`.
- Phase Plot — `plot_phase_plot(phase_df)`
  - Input: `[X0, ratio, X_final]` from `phase_sweep_df`.
  - Heatmap of `X*` across initial adoption `X0` and ratio `a_I/b`; overlays the threshold curve `X = 1/ratio`.
- Fan Chart (baseline vs subsidy)
  - Prefer the DataFrame variant in `ev_plotting.py` (ensures `plots/` outputs). The legacy `plot_intervention_fanchart` inside `ev_experiments.py` saves to current directory.

## Setting Up Different Experiments
- Customize `scenario_kwargs` when calling runners:
  - Common fields:
    - `ratio` or `a0` (use `ratio` to compute `a0 = ratio*b - beta_I*I0`)
    - `beta_I`, `b`, `g_I`, `I0`
    - `network_type`, `n_nodes`, `p`, `m`
    - `collect`, `X0_frac`, `init_method`
  - Example:
    - ```python
      scenario = {
          "ratio": 2.3,
          "beta_I": 2.0, "b": 1.0, "g_I": 0.10, "I0": 0.05,
          "network_type": "BA", "n_nodes": 300, "m": 2,
          "collect": True, "X0_frac": 0.40, "init_method": "random",
      }
      ```
- Switch decision rules:
  - Set `strategy_choice_func="imitate"` (default) or `"logit"` and adjust `tau`.
- Use policies:
  - Pass a closure directly to `run_timeseries_trial` or use policy spec dicts for parallel workers as shown above.
- Build DataFrames and plot:
  - Traces: `traces_df = traces_to_long_df(baseline_X, subsidy_X)` then `plot_spaghetti(...)` and `plot_density(...)`.
  - Ratio sweep: `sweep_df = ratio_sweep_df(...)` then `plot_ratio_sweep(sweep_df)`.
  - Phase plot: `phase_df = phase_sweep_df(...)` then `plot_phase_plot(phase_df)`.

## Reproduce the Included Demo
- From `Code/Assignment 3`, run:
  - `python ev_experiments.py`
- What it does:
  - Generates baseline vs subsidy fan chart (legacy path).
  - Builds phase plot DataFrame and saves heatmap to `plots/ev_phase_plot.png`.
  - Runs many trials to generate spaghetti and time-evolving density plots in `plots/`.
  - Computes ratio sweep and saves line plot to `plots/`.

## Tips
- Performance
  - Use `max_workers > 1` with `backend="process"` in `phase_sweep_df` for speed on large sweeps.
  - Reduce `n_nodes`, `batch_size`, or horizon `T` for quicker explorations.
- Robustness
  - Try both ER and BA networks for the same scenario.
  - Use `init_method="degree"` to seed hubs and test tipping behavior.
- Consistency
  - Prefer plotting via `ev_plotting.py` to ensure all images save under `plots/`.

## File Reference
- Core APIs (ev_core.py)
  - `EVStagHuntModel(...)`: construct model with parameters above.
  - `set_initial_adopters(model, X0_frac, method="random"|"degree", seed=None, high=True)`.
  - `final_mean_adoption_vs_ratio(...)` → `np.ndarray` of means.
  - `phase_sweep_X0_vs_ratio(...)` → `np.ndarray` matrix of means.
- Experiments (ev_experiments.py)
  - `run_timeseries_trial(...)` → `(X, I, df)`.
  - `collect_intervention_trials(...)` → raw trajectories + summaries.
  - `traces_to_long_df(...)`, `ratio_sweep_df(...)`, `phase_sweep_df(...)` → DataFrames for plotting.
  - `run_intervention_example(...)` → `(baseline_df, subsidy_df, img_path)`.
  - `main()` → demo pipeline that saves plots.
- Plotting (ev_plotting.py)
  - `plot_fanchart(...)`, `plot_spaghetti(...)`, `plot_density(...)`, `plot_ratio_sweep(...)`, `plot_phase_plot(...)` → return saved image path.

---
If you need an additional plot style, experiment type, or policy, add a new DataFrame builder in `ev_experiments.py` and a companion function in `ev_plotting.py` to keep computation and visualization cleanly separated.