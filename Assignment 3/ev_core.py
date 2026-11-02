"""
Core EV Stag Hunt model components.

Contains the Mesa agent and model classes, plus basic strategy
selection helpers and initial adopter utilities. This file is
free of experiment-specific policies and plotting.

The model implements the EV Stag Hunt game on a network, where agents
choose between adopting EVs ("C") and defection ("D") based on their
payoffs. The model tracks agent strategies, payoffs, and network
structure over time. Cooperation means adopting EVs, while defection
means staying with Internal Combustion Engines (ICE).

Agents play the game on a network, where each node represents an agent.
Agents interact with their neighbours, compute payoffs, and update their
strategies based on the selected rule. The model tracks agent strategies,
payoffs, and network structure over time.

Infrastructure growth is modeled as a function of the current
infrastructure level, where each step of infrastructure growth
increases the effective coordination payoff for EV adoption. Infrastructure grows faster
with higher levels of adoption.

The equation for growth is:
I(t+1) = I(t) + g_I * I(t)
"""

from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import random
import os
from typing import Iterable, List, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

####################################
# Strategy selection helpers
#
# We provide two different ways agents can choose their strategy:
# - `choose_strategy_imitate`: choose strategy of the highest-payoff neighbour (including self).
# - `choose_strategy_logit`: choose strategy using logit / softmax choice.
#
####################################
def choose_strategy_imitate(agent, neighbors):
    """Choose strategy of the highest-payoff neighbour (including self)."""
    candidates = neighbors + [agent]
    best = max(candidates, key=lambda a: a.payoff)
    return best.strategy


def choose_strategy_logit(agent, neighbors, a_I, b, tau):
    """Choose strategy using logit / softmax choice.

    Parameters
    - agent: the agent choosing a strategy
    - neighbors: list of neighbour agents
    - a_I: effective coordination payoff given current infrastructure
    - b: defection payoff
    - tau: temperature parameter for softmax
    """
    # compute expected payoffs for C and D
    pi_C = 0.0
    pi_D = 0.0
    for other in neighbors:
        s_j = other.strategy
        if s_j == "C":
            pi_C += a_I
            pi_D += b
        else:
            pi_C += 0.0
            pi_D += b

    # softmax choice
    denom = np.exp(pi_C / tau) + np.exp(pi_D / tau)
    P_C = np.exp(pi_C / tau) / denom if denom > 0 else 0.5
    return "C" if random.random() < P_C else "D"



####################################
# Agent class
#
# The EVAgent class implements the single agent at a graph node.
#
# Attributes
# - strategy: "C" (adopt EV) or "D" (defect / ICE)
# - payoff: accumulated payoff from interactions with neighbours
# - next_strategy: strategy chosen for the next time step
####################################
class EVAgent(Agent):
    """Single agent at a graph node.

    Attributes
    - strategy: "C" (adopt EV) or "D" (defect / ICE)
    - payoff: accumulated payoff from interactions with neighbours
    - next_strategy: strategy chosen for the next time step
    """

    def __init__(self, unique_id, model, init_strategy="D"):
        super().__init__(unique_id, model)
        self.strategy = init_strategy
        self.payoff = 0.0
        self.next_strategy = init_strategy

    def step(self):
        """Compute payoff from interactions with neighbours.

        Stag Hunt payoff rules:
        - C vs C: `a_I` (coordination enhanced by infrastructure)
        - C vs D: 0
        - D vs C: `b`
        - D vs D: `b`
        """
        I = self.model.infrastructure
        a0 = self.model.a0
        beta_I = self.model.beta_I
        b = self.model.b
        a_I = a0 + beta_I * I

        neighbor_agents = []
        for nbr in self.model.G.neighbors(self.pos):
            neighbor_agents.extend(self.model.grid.get_cell_list_contents([nbr]))
        if not neighbor_agents:
            self.payoff = 0.0
            return

        payoff = 0.0
        for other in neighbor_agents:
            s_i = self.strategy
            s_j = other.strategy
            if s_i == "C" and s_j == "C":
                payoff += a_I
            elif s_i == "C" and s_j == "D":
                payoff += 0.0
            elif s_i == "D" and s_j == "C":
                payoff += b
            else:
                payoff += b
        self.payoff = payoff

    ####################################
    # Advance method
    #
    # The advance method updates the agent's strategy based on the selected rule.
    #
    # Parameters
    # - strategy_choice_func: the strategy selection function to use ("imitate" or "logit")
    ####################################
    def advance(self, strategy_choice_func="imitate"):
        """Update next_strategy using the selected rule.

        If called without an explicit rule, read `self.model.strategy_choice_func`.
        Commit `self.strategy = self.next_strategy` for synchronous updates.
        """
        func = strategy_choice_func if strategy_choice_func is not None else getattr(self.model, "strategy_choice_func", "imitate")

        neighbor_agents = []
        for nbr in self.model.G.neighbors(self.pos):
            neighbor_agents.extend(self.model.grid.get_cell_list_contents([nbr]))

        if func == "imitate":
            self.next_strategy = choose_strategy_imitate(self, neighbor_agents)
        elif func == "logit":
            a_I = self.model.a0 + self.model.beta_I * self.model.infrastructure
            self.next_strategy = choose_strategy_logit(self, neighbor_agents, a_I, self.model.b, getattr(self.model, "tau", 1.0))
        else:
            raise ValueError(f"Unknown strategy choice function: {func}")

        self.strategy = self.next_strategy

    ####################################
    # Model class
    #
    # The EVStagHuntModel class implements the Mesa model for EV Stag Hunt on a network.
    #
    # Parameters
    # - initial_ev: number of initial EV nodes
    # - a0: base payoff for EV adoption
    # - beta_I: payoff enhancement factor for EV adoption
    # - b: payoff for ICE defection
    # - g_I: infrastructure growth rate
    # - I0: initial infrastructure level
    # - seed: random seed for reproducibility
    # - network_type: type of network to generate ("random" or "BA")
    # - n_nodes: number of nodes in the network
    # - p: probability of edge creation in random network
    # - m: number of edges to attach to new node in BA network
    # - collect: whether to collect agent and model-level data
    # - strategy_choice_func: strategy selection function ("imitate" or "logit")
    # - tau: temperature parameter for softmax choice (only used with "logit")
    ####################################
#
class EVStagHuntModel(Model):
    """Mesa model for EV Stag Hunt on a network."""

    def __init__(
        self,
        initial_ev=10,
        a0=2.0,
        beta_I=3.0,
        b=1.0,
        g_I=0.1,
        I0=0.05,
        seed=None,
        network_type="random",
        n_nodes=100,
        p=0.05,
        m=2,
        collect=True,
        strategy_choice_func: str = "imitate",
        tau: float = 1.0,
    ):
        super().__init__(seed=seed)

        # Build graph
        if network_type == "BA":
            G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
        else:
            G = nx.erdos_renyi_graph(n_nodes, p, seed=seed)
        self.G = G
        self.grid = NetworkGrid(G)
        self.schedule = SimultaneousActivation(self)

        # parameters
        self.a0 = a0
        self.beta_I = beta_I
        self.b = b
        self.g_I = g_I
        self.infrastructure = I0
        self.step_count = 0
        self.strategy_choice_func = strategy_choice_func
        self.tau = tau

        # initialize node attribute for agent reference
        for n in self.G.nodes:
            self.G.nodes[n]["agent"] = []

        # choose initial EV nodes
        total_nodes = self.G.number_of_nodes()
        k_ev = max(0, min(initial_ev, total_nodes))
        ev_nodes = set(self.random.sample(list(self.G.nodes), k_ev))

        # create one agent per node
        uid = 0
        for node in self.G.nodes:
            init_strategy = "C" if node in ev_nodes else "D"
            agent = EVAgent(uid, self, init_strategy)
            uid += 1
            self.schedule.add(agent)
            self.grid.place_agent(agent, node)

        self.datacollector = None
        if collect:
            self.datacollector = DataCollector(
                model_reporters={
                    "X": self.get_adoption_fraction,
                    "I": lambda m: m.infrastructure,
                },
                agent_reporters={"strategy": "strategy", "payoff": "payoff"},
            )

    def get_adoption_fraction(self):
        agents = self.schedule.agents
        if not agents:
            return 0.0
        return sum(1 for a in agents if a.strategy == "C") / len(agents)

    # ####################
    # Model step function
    #
    # The step function advances the model by one time step.
    # It first advances all agents, then computes the adoption fraction and infrastructure level.
    # The infrastructure level is updated based on the adoption fraction and the infrastructure growth rate.
    # The updated infrastructure level is clipped to the interval [0, 1].
    # Finally, if data collection is enabled, the model and agent data are collected.
    #######################
    def step(self): 
        self.schedule.step() # advance all agents
        X = self.get_adoption_fraction() # compute adoption fraction after all agents have advanced
        I = self.infrastructure # infrastructure level before this step
        dI = self.g_I * (X - I) # infrastructure growth rate, impacted by adoption fraction
        self.infrastructure = float(min(1.0, max(0.0, I + dI))) # clip infrastructure level to [0, 1]
        if self.datacollector is not None:
            self.datacollector.collect(self) # collect data at the end of each step
        self.step_count += 1 # increment step count after data collection

#########################
#
# Set initial adopters
# 
# Parameters
# - model: the EVStagHuntModel instance
# - X0_frac: fraction of agents to initially choose EV adoption
# - method: method to choose initial adopters ("random" or "degree")
# - seed: random seed for reproducibility
# - high: whether to choose high or low degree nodes for "degree" method
###########################
def set_initial_adopters(model, X0_frac, method="random", seed=None, high=True):
    """Set a fraction of agents to EV adopters using different heuristics."""
    rng = np.random.default_rng(seed)
    agents = model.schedule.agents
    n = len(agents)
    k = int(round(X0_frac * n))

    for a in agents:
        a.strategy = "D"

    if k <= 0:
        return

    if method == "random":
        idx = rng.choice(n, size=k, replace=False)
        for i in idx:
            agents[i].strategy = "C"
        return

    if method == "degree":
        deg = dict(model.G.degree())
        ordered_nodes = sorted(deg.keys(), key=lambda u: deg[u], reverse=high)
        chosen = set(ordered_nodes[:k])
        for a in agents:
            if a.unique_id in chosen:
                a.strategy = "C"
        return

    raise ValueError(f"Unknown method: {method}")


# -----------------------------
# Ratio sweep helpers (computation-only)
# -----------------------------
#########################
#
# Run a single network trial
# 
# Parameters
# - X0_frac: fraction of agents to initially choose EV adoption
# - ratio: payoff ratio between EV and DC agents (a0 = ratio*b - beta_I*I0)
# - I0: initial infrastructure level
# - beta_I: cost of EV adoption relative to DC (beta_I*I0)
# - b: payoff of EV (b)
# - g_I: infrastructure growth rate (g_I)
# - T: number of time steps to run
# - network_type: type of network to generate ("random" or "BA")
# - n_nodes: number of nodes in the network
# - p: probability of edge creation in random network
# - m: number of edges to attach from a new node to existing nodes in BA network
# - seed: random seed for reproducibility
# - tol: tolerance for convergence check (default: 1e-3)
# - patience: number of steps to wait for convergence (default: 30)

def run_network_trial(
    X0_frac: float,
    ratio: float,
    *,
    I0: float = 0.05,
    beta_I: float = 2.0,
    b: float = 1.0,
    g_I: float = 0.05,
    T: int = 200,
    network_type: str = "random",
    n_nodes: int = 120,
    p: float = 0.05,
    m: int = 2,
    seed: int | None = None,
    tol: float = 1e-3,
    patience: int = 30,
    collect: bool = False,
    strategy_choice_func: str = "imitate",
    tau: float = 1.0,
) -> float:
    """Run a single realisation and return final adoption fraction.

    Preserves the intended initial payoff ratio via a0 = ratio*b - beta_I*I0.
    Includes basic stability-based early stopping.
    """
    initial_ev = int(round(X0_frac * n_nodes))
    a0 = ratio * b - beta_I * I0

    model = EVStagHuntModel(
        initial_ev=initial_ev,
        a0=a0,
        beta_I=beta_I,
        b=b,
        g_I=g_I,
        I0=I0,
        seed=seed,
        network_type=network_type,
        n_nodes=n_nodes,
        p=p,
        m=m,
        collect=collect,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )

    stable_steps = 0
    prev_X = None
    prev_I = None
    for _ in range(T):
        model.step()
        X = model.get_adoption_fraction()
        I = model.infrastructure
        if prev_X is not None and prev_I is not None:
            if abs(X - prev_X) < tol and abs(I - prev_I) < tol:
                stable_steps += 1
            else:
                stable_steps = 0
        prev_X, prev_I = X, I
        if X in (0.0, 1.0) and stable_steps >= 10:
            break
        if stable_steps >= patience:
            break

    return model.get_adoption_fraction()

#########################
#
# Compute final mean adoption fraction vs ratio
# 
##########################
def final_mean_adoption_vs_ratio(
    X0_frac: float,
    ratio_values: Iterable[float],
    *,
    I0: float = 0.05,
    beta_I: float = 2.0,
    b: float = 1.0,
    g_I: float = 0.05,
    T: int = 200,
    network_type: str = "random",
    n_nodes: int = 120,
    p: float = 0.05,
    m: int = 2,
    batch_size: int = 16,
    init_noise_I: float = 0.04,
    strategy_choice_func: str = "imitate",
    tau: float = 1.0,
) -> np.ndarray:
    """Compute mean final adoption across a sweep of ratio values.

    For each ratio, average over `batch_size` trials with jittered `I0` and seeds.
    Returns a numpy array of means aligned with `ratio_values` order.
    """
    ratios = list(ratio_values)
    means: List[float] = []
    for ratio in ratios:
        finals: List[float] = []
        for _ in range(batch_size):
            I0_j = float(np.clip(np.random.normal(loc=I0, scale=init_noise_I), 0.0, 1.0))
            seed_j = np.random.randint(0, 2**31 - 1)
            x_star = run_network_trial(
                X0_frac,
                ratio,
                I0=I0_j,
                beta_I=beta_I,
                b=b,
                g_I=g_I,
                T=T,
                network_type=network_type,
                n_nodes=n_nodes,
                p=p,
                m=m,
                seed=seed_j,
                collect=False,
                strategy_choice_func=strategy_choice_func,
                tau=tau,
            )
            finals.append(x_star)
        means.append(float(np.mean(finals)))
    return np.asarray(means, dtype=float)

#########################
#
# Compute heatmap row for a fixed ratio
# 
##########################
def _row_for_ratio_task(args: Dict) -> np.ndarray:
    """Top-level worker to compute one heatmap row for a fixed ratio.

    Returns an array of mean final adoption across provided X0_values.
    """
    ratio = args["ratio"]
    X0_values = args["X0_values"]
    I0 = args["I0"]
    beta_I = args["beta_I"]
    b = args["b"]
    g_I = args["g_I"]
    T = args["T"]
    network_type = args["network_type"]
    n_nodes = args["n_nodes"]
    p = args["p"]
    m = args["m"]
    batch_size = args["batch_size"]
    init_noise_I = args["init_noise_I"]
    strategy_choice_func = args["strategy_choice_func"]
    tau = args["tau"]

    row = np.empty(len(X0_values), dtype=float)
    for j, X0 in enumerate(X0_values):
        finals: List[float] = []
        for _ in range(batch_size):
            I0_j = float(np.clip(np.random.normal(loc=I0, scale=init_noise_I), 0.0, 1.0))
            seed_j = np.random.randint(0, 2**31 - 1)
            x_star = run_network_trial(
                X0_frac=X0,
                ratio=ratio,
                I0=I0_j,
                beta_I=beta_I,
                b=b,
                g_I=g_I,
                T=T,
                network_type=network_type,
                n_nodes=n_nodes,
                p=p,
                m=m,
                seed=seed_j,
                collect=False,
                strategy_choice_func=strategy_choice_func,
                tau=tau,
            )
            finals.append(x_star)
        row[j] = float(np.mean(finals))
    return row

    
#########################
#
# Compute heatmap matrix for phase sweep
# 
##########################
def phase_sweep_X0_vs_ratio(
    X0_values: Iterable[float],
    ratio_values: Iterable[float],
    *,
    I0: float = 0.05,
    beta_I: float = 2.0,
    b: float = 1.0,
    g_I: float = 0.05,
    T: int = 250,
    network_type: str = "BA",
    n_nodes: int = 120,
    p: float = 0.05,
    m: int = 2,
    batch_size: int = 16,
    init_noise_I: float = 0.04,
    strategy_choice_func: str = "logit",
    tau: float = 1.0,
    max_workers: int | None = None,
    backend: str = "process",
) -> np.ndarray:
    """Compute a heatmap matrix of mean final adoption X* over (X0, ratio).

    Returns an array of shape (len(ratio_values), len(X0_values)) aligned with
    the provided orders. Rows correspond to ratios; columns to X0 values.
    """
    X0_values = list(X0_values)
    ratio_values = list(ratio_values)
    X_final = np.zeros((len(ratio_values), len(X0_values)), dtype=float)

    # Prepare tasks per ratio
    tasks: List[Dict] = []
    for ratio in ratio_values:
        tasks.append({
            "ratio": ratio,
            "X0_values": X0_values,
            "I0": I0,
            "beta_I": beta_I,
            "b": b,
            "g_I": g_I,
            "T": T,
            "network_type": network_type,
            "n_nodes": n_nodes,
            "p": p,
            "m": m,
            "batch_size": batch_size,
            "init_noise_I": init_noise_I,
            "strategy_choice_func": strategy_choice_func,
            "tau": tau,
        })

    if max_workers is None:
        try:
            max_workers = os.cpu_count() or 1
        except Exception:
            max_workers = 1

    Executor = ProcessPoolExecutor if backend == "process" and max_workers > 1 else ThreadPoolExecutor
    if max_workers > 1:
        with Executor(max_workers=max_workers) as ex:
            futures = [ex.submit(_row_for_ratio_task, args) for args in tasks]
            for i, fut in enumerate(futures):
                row = fut.result()
                X_final[i, :] = row
    else:
        for i, args in enumerate(tasks):
            row = _row_for_ratio_task(args)
            X_final[i, :] = row

    return X_final


# -----------------------------
# Baseline Simulation & X0 vs I0 Sweep (Assignment Refactored)
# -----------------------------

def run_simulation_trial(
    X0_frac: float = 0.1,
    *,
    a0: float = 1.0,
    beta_I: float = 2.5,
    b: float = 1.0,
    g_I: float = 0.1,
    I0: float = 0.1,
    T: int = 150,
    network_type: str = "random",
    n_nodes: int = 200,
    p: float = 0.03,
    m: int = 3,
    seed: int | None = None,
    init_method: str = "random",
    high: bool = True,
    strategy_choice_func: str = "imitate",
    tau: float = 1.0,
    policy = None,
) -> float:
    """Run a single baseline simulation trial and return final adoption fraction X(T)."""
    model = EVStagHuntModel(
        initial_ev=0,
        a0=a0,
        beta_I=beta_I,
        b=b,
        g_I=g_I,
        I0=I0,
        seed=seed,
        network_type=network_type,
        n_nodes=n_nodes,
        p=p,
        m=m,
        collect=False,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )
    set_initial_adopters(model, X0_frac, method=init_method, seed=seed, high=high)
    for step in range(T):
        if policy is not None:
            policy(model, step)
        model.step()
        if policy is None and strategy_choice_func == "imitate":
            x = model.get_adoption_fraction()
            if x == 0.0 or x == 1.0:
                return x
    return model.get_adoption_fraction()


def _row_for_I0_task(args: Dict) -> np.ndarray:
    """Worker function to compute one heatmap row for a fixed initial infrastructure I0."""
    I0 = args["I0"]
    X0_values = args["X0_values"]
    a0 = args["a0"]
    beta_I = args["beta_I"]
    b = args["b"]
    g_I = args["g_I"]
    T = args["T"]
    network_type = args["network_type"]
    n_nodes = args["n_nodes"]
    p = args["p"]
    m = args["m"]
    n_seeds = args["n_seeds"]
    strategy_choice_func = args["strategy_choice_func"]
    tau = args["tau"]
    init_method = args.get("init_method", "random")

    row = np.empty(len(X0_values), dtype=float)
    for j, X0 in enumerate(X0_values):
        finals: List[float] = []
        for s in range(n_seeds):
            seed_s = int(abs(hash((round(float(X0), 4), round(float(I0), 4), s))) % (2**31 - 1))
            x_star = run_simulation_trial(
                X0_frac=X0,
                a0=a0,
                beta_I=beta_I,
                b=b,
                g_I=g_I,
                I0=I0,
                T=T,
                network_type=network_type,
                n_nodes=n_nodes,
                p=p,
                m=m,
                seed=seed_s,
                init_method=init_method,
                strategy_choice_func=strategy_choice_func,
                tau=tau,
            )
            finals.append(x_star)
        row[j] = float(np.mean(finals))
    return row


def phase_sweep_X0_vs_I0(
    X0_values: Iterable[float],
    I0_values: Iterable[float],
    *,
    a0: float = 1.0,
    beta_I: float = 2.5,
    b: float = 1.0,
    g_I: float = 0.1,
    T: int = 150,
    network_type: str = "random",
    n_nodes: int = 200,
    p: float = 0.03,
    m: int = 3,
    n_seeds: int = 10,
    strategy_choice_func: str = "imitate",
    tau: float = 1.0,
    init_method: str = "random",
    max_workers: int | None = None,
    backend: str = "process",
) -> np.ndarray:
    """Compute a 2D matrix of mean final adoption X* over (X0, I0).

    Returns an array of shape (len(I0_values), len(X0_values)).
    Rows correspond to I0 values (y-axis); columns correspond to X0 values (x-axis).
    """
    X0_values = list(X0_values)
    I0_values = list(I0_values)
    X_final = np.zeros((len(I0_values), len(X0_values)), dtype=float)

    tasks: List[Dict] = []
    for I0 in I0_values:
        tasks.append({
            "I0": float(I0),
            "X0_values": X0_values,
            "a0": a0,
            "beta_I": beta_I,
            "b": b,
            "g_I": g_I,
            "T": T,
            "network_type": network_type,
            "n_nodes": n_nodes,
            "p": p,
            "m": m,
            "n_seeds": n_seeds,
            "strategy_choice_func": strategy_choice_func,
            "tau": tau,
            "init_method": init_method,
        })

    if max_workers is None:
        try:
            max_workers = os.cpu_count() or 1
        except Exception:
            max_workers = 1

    Executor = ProcessPoolExecutor if backend == "process" and max_workers > 1 else ThreadPoolExecutor
    if max_workers > 1:
        with Executor(max_workers=max_workers) as ex:
            futures = [ex.submit(_row_for_I0_task, args) for args in tasks]
            for i, fut in enumerate(futures):
                X_final[i, :] = fut.result()
    else:
        for i, args in enumerate(tasks):
            X_final[i, :] = _row_for_I0_task(args)

    return X_final