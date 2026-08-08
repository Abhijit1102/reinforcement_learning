"""
GridWorld — Value Iteration
============================

Bellman Optimality Equation:

    V*(s) = max_a  Σ_s',r p(s',r|s,a) [ r + γ V*(s') ]

Since transitions here are deterministic, this simplifies to:

    V*(s) = max_a [ r(s,a) + γ V*(s') ]

Value Iteration applies this update as a single sweep (instead of separately
doing policy evaluation + policy improvement like in Policy Iteration).
Once V* has converged, the optimal policy is extracted greedily:

    π*(s) = argmax_a [ r(s,a) + γ V*(s') ]
"""

import time
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

# ----------------------------------------------------------------------------
# GridWorld setup (same world as the policy-evaluation mini-project)
# ----------------------------------------------------------------------------
GRID_SIZE = 4
GAMMA = 1.0
THETA = 1e-4
TERMINAL_STATES = {(0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)}
ACTIONS = {
    "UP":    (-1, 0),
    "DOWN":  (1, 0),
    "LEFT":  (0, -1),
    "RIGHT": (0, 1),
}
ARROWS = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}

console = Console()


def all_states():
    return [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]


def next_state(state, action):
    if state in TERMINAL_STATES:
        return state
    dr, dc = ACTIONS[action]
    r, c = state
    nr, nc = r + dr, c + dc
    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
        return (nr, nc)
    return state


def reward(state, next_s):
    if state in TERMINAL_STATES:
        return 0
    return -1


def make_value_table(V, iteration, delta):
    table = Table(
        title=f"Value Iteration — sweep {iteration}   (Δ = {delta:.5f})",
        show_header=False,
        show_lines=True,
    )
    for _ in range(GRID_SIZE):
        table.add_column(justify="center", width=9)
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            if (r, c) in TERMINAL_STATES:
                row.append("[bold green]TERMINAL[/bold green]")
            else:
                row.append(f"{V[(r, c)]:6.2f}")
        table.add_row(*row)
    return table


def make_policy_table(policy):
    table = Table(title="Optimal Policy π*", show_header=False, show_lines=True)
    for _ in range(GRID_SIZE):
        table.add_column(justify="center", width=9)
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            s = (r, c)
            if s in TERMINAL_STATES:
                row.append("[bold green]TERMINAL[/bold green]")
            else:
                row.append(f"[bold cyan]{ARROWS[policy[s]]}[/bold cyan]")
        table.add_row(*row)
    return table


def value_iteration_step(V):
    """One sweep of the Bellman Optimality update: V*(s) = max_a [r + γV(s')]"""
    V_new = {}
    for s in all_states():
        if s in TERMINAL_STATES:
            V_new[s] = 0.0
            continue
        action_values = []
        for action in ACTIONS:
            s_next = next_state(s, action)
            r = reward(s, s_next)
            action_values.append(r + GAMMA * V[s_next])
        V_new[s] = max(action_values)
    return V_new


def extract_policy(V):
    """Greedy policy from converged V*: π*(s) = argmax_a [r + γV*(s')]"""
    policy = {}
    for s in all_states():
        if s in TERMINAL_STATES:
            continue
        best_action, best_value = None, float("-inf")
        for action in ACTIONS:
            s_next = next_state(s, action)
            r = reward(s, s_next)
            value = r + GAMMA * V[s_next]
            if value > best_value:
                best_value = value
                best_action = action
        policy[s] = best_action
    return policy


def run_value_iteration(animate=True, delay=0.4):
    V = {s: 0.0 for s in all_states()}
    iteration = 0

    if animate:
        with Live(make_value_table(V, iteration, 0.0), console=console, refresh_per_second=4) as live:
            while True:
                V_new = value_iteration_step(V)
                delta = max(abs(V_new[s] - V[s]) for s in all_states())
                iteration += 1
                V = V_new
                live.update(make_value_table(V, iteration, delta))
                time.sleep(delay)
                if delta < THETA:
                    break
    else:
        while True:
            V_new = value_iteration_step(V)
            delta = max(abs(V_new[s] - V[s]) for s in all_states())
            iteration += 1
            V = V_new
            if delta < THETA:
                break

    return V, iteration


if __name__ == "__main__":
    console.print(
        Panel(
            "[bold]Value Iteration — Bellman Optimality Equation[/bold]\n"
            "4x4 GridWorld | γ = 1.0 | reward = -1/step\n"
            "V*(s) = max_a [ r + γ·V*(s') ]",
            border_style="magenta",
        )
    )

    V_star, num_iterations = run_value_iteration(animate=True, delay=0.35)

    console.print(f"\n[bold green]Converged after {num_iterations} sweeps "
                  f"(Δ < {THETA})[/bold green]\n")

    console.print(make_value_table(V_star, num_iterations, 0.0))

    optimal_policy = extract_policy(V_star)
    console.print()
    console.print(make_policy_table(optimal_policy))
