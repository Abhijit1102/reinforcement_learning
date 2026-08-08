"""
GridWorld — Iterative Policy Evaluation via the Bellman Expectation Equation
=============================================================================

Bellman Expectation Equation for state-value function under policy π:

    V(s) = Σ_a π(a|s) * Σ_s',r p(s',r|s,a) * [ r + γ * V(s') ]

For this GridWorld:
  - Actions are deterministic (each action moves exactly one cell, or bounces
    off a wall and stays in place), so the inner sum over (s', r) collapses
    to a single term.
  - The policy π is the uniform random policy: π(a|s) = 1/4 for all 4 actions.
  - Reward r = -1 for every transition until a terminal state is reached.
  - γ (gamma) = 1.0 (episodic, undiscounted).

So the equation used in code each sweep is:

    V(s) = (1/4) * Σ_a [ r(s,a) + γ * V(s') ]
"""

import time
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

# ----------------------------------------------------------------------------
# GridWorld setup
# ----------------------------------------------------------------------------
GRID_SIZE = 4
GAMMA = 1.0
THETA = 1e-4                     # convergence threshold
TERMINAL_STATES = {(0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)}
ACTIONS = {
    "UP":    (-1, 0),
    "DOWN":  (1, 0),
    "LEFT":  (0, -1),
    "RIGHT": (0, 1),
}
ACTION_PROB = 1 / len(ACTIONS)   # uniform random policy

console = Console()


def all_states():
    return [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]


def next_state(state, action):
    """Deterministic transition; bounce off walls (stay in place)."""
    if state in TERMINAL_STATES:
        return state
    dr, dc = ACTIONS[action]
    r, c = state
    nr, nc = r + dr, c + dc
    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
        return (nr, nc)
    return state  # hit a wall, no movement


def reward(state, next_s):
    """-1 per step, 0 once already terminal (no further reward accrues)."""
    if state in TERMINAL_STATES:
        return 0
    return -1


def make_table(V, iteration, delta):
    table = Table(
        title=f"Iteration {iteration}   (Δ = {delta:.5f})",
        show_header=False,
        show_lines=True,
    )
    for _ in range(GRID_SIZE):
        table.add_column(justify="center", width=9)

    for r in range(GRID_SIZE):
        row_cells = []
        for c in range(GRID_SIZE):
            if (r, c) in TERMINAL_STATES:
                row_cells.append("[bold green]TERMINAL[/bold green]")
            else:
                row_cells.append(f"{V[(r, c)]:6.2f}")
        table.add_row(*row_cells)
    return table


def bellman_expectation_update(V):
    """
    One synchronous sweep applying the Bellman Expectation Equation to every
    state:  V_new(s) = Σ_a π(a|s) [ r + γ V(s') ]
    """
    V_new = {}
    for s in all_states():
        if s in TERMINAL_STATES:
            V_new[s] = 0.0
            continue

        value = 0.0
        for action in ACTIONS:
            s_next = next_state(s, action)
            r = reward(s, s_next)
            value += ACTION_PROB * (r + GAMMA * V[s_next])
        V_new[s] = value
    return V_new


def iterative_policy_evaluation(animate=True, delay=0.6):
    V = {s: 0.0 for s in all_states()}
    iteration = 0

    if animate:
        with Live(make_table(V, iteration, delta=0.0), console=console, refresh_per_second=4) as live:
            while True:
                V_new = bellman_expectation_update(V)
                delta = max(abs(V_new[s] - V[s]) for s in all_states())
                iteration += 1
                V = V_new
                live.update(make_table(V, iteration, delta))
                time.sleep(delay)
                if delta < THETA:
                    break
    else:
        while True:
            V_new = bellman_expectation_update(V)
            delta = max(abs(V_new[s] - V[s]) for s in all_states())
            iteration += 1
            V = V_new
            if delta < THETA:
                break

    return V, iteration


if __name__ == "__main__":
    console.print(
        Panel(
            "[bold]Bellman Expectation Equation — Iterative Policy Evaluation[/bold]\n"
            "4x4 GridWorld | Uniform random policy | γ = 1.0 | reward = -1/step\n"
            "V(s) = Σ_a π(a|s) · [ r + γ·V(s') ]",
            border_style="cyan",
        )
    )

    final_V, num_iterations = iterative_policy_evaluation(animate=True, delay=0.4)

    console.print(f"\n[bold green]Converged after {num_iterations} iterations "
                  f"(Δ < {THETA})[/bold green]\n")
    console.print(make_table(final_V, num_iterations, delta=0.0))
