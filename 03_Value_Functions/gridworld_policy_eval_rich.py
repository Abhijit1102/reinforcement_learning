"""
Policy Evaluation for GridWorld under a UNIFORM RANDOM POLICY, using rich.

Computes V(s) = sum_a pi(a|s) * [R(s,a,s') + gamma * V(s')]
via iterative sweeps (brute-force / "in-place" policy evaluation),
displaying the value table after every sweep until it converges.

Run:
    pip install rich
    python gridworld_policy_eval_rich.py
"""

import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

# ----------------------------- MDP definition ------------------------------ #

GRID_SIZE = 3
GOAL = (2, 2)
STATES = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
ACTIONS = {
    "UP":    (-1, 0),
    "DOWN":  (1, 0),
    "LEFT":  (0, -1),
    "RIGHT": (0, 1),
}
STEP_REWARD = -0.1
GOAL_REWARD = 1.0
GAMMA = 0.9
THETA = 1e-6          # convergence threshold
POLICY_PROB = 0.25     # uniform random policy: P(a|s) = 1/4 for every action


def build_mdp():
    """mdp[(s,a)] = (next_state, reward)  -- deterministic transitions."""
    mdp = {}
    for s in STATES:
        if s == GOAL:
            continue
        r, c = s
        for a, (dr, dc) in ACTIONS.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                nr, nc = r, c  # wall bump -> stay
            ns = (nr, nc)
            reward = GOAL_REWARD if ns == GOAL else STEP_REWARD
            mdp[(s, a)] = (ns, reward)
    return mdp


MDP = build_mdp()

# ------------------------------ Policy evaluation --------------------------- #

def policy_eval_sweep(V):
    """One synchronous sweep of the Bellman expectation update. Returns (new_V, max_delta)."""
    new_V = dict(V)
    max_delta = 0.0
    for s in STATES:
        if s == GOAL:
            new_V[s] = 0.0
            continue
        total = 0.0
        for a in ACTIONS:
            ns, reward = MDP[(s, a)]
            total += POLICY_PROB * (reward + GAMMA * V[ns])
        new_V[s] = total
        max_delta = max(max_delta, abs(new_V[s] - V[s]))
    return new_V, max_delta


def render_value_table(V, iteration, delta):
    table = Table(
        title=f"V(s) under Random Policy  |  Sweep {iteration}  |  max Δ = {delta:.6f}",
        show_header=False,
        show_lines=True,
    )
    for _ in range(GRID_SIZE):
        table.add_column(justify="center", width=10)
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            s = (r, c)
            label = "GOAL\n0.0000" if s == GOAL else f"{s}\n{V[s]:+.4f}"
            style = "bold yellow on black" if s == GOAL else (
                "bold green" if V[s] > 0 else "white"
            )
            row.append(f"[{style}]{label}[/{style}]")
        table.add_row(*row)
    return table


def main():
    console = Console()
    console.print(Panel.fit("GridWorld Policy Evaluation (Uniform Random Policy)", style="bold magenta"))
    console.print(f"γ = {GAMMA},  θ (convergence threshold) = {THETA},  policy: 25% each action\n")

    V = {s: 0.0 for s in STATES}  # initialize all values to 0
    iteration = 0

    with Live(render_value_table(V, iteration, float("inf")), console=console, refresh_per_second=10) as live:
        while True:
            new_V, delta = policy_eval_sweep(V)
            iteration += 1
            V = new_V
            live.update(render_value_table(V, iteration, delta))
            time.sleep(0.3)  # slow down so each sweep is visible; set to 0 for instant
            if delta < THETA:
                break

    console.print(f"\n[bold green]Converged[/bold green] after {iteration} sweeps.\n")

    final_table = Table(title="Final V(s) under Random Policy", show_header=True)
    final_table.add_column("State")
    final_table.add_column("V(s)", justify="right")
    for s in STATES:
        final_table.add_row(str(s) if s != GOAL else f"{s} (GOAL)", f"{V[s]:+.4f}")
    console.print(final_table)


if __name__ == "__main__":
    main()
