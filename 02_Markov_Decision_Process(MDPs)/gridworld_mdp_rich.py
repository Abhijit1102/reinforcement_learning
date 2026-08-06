"""
GridWorld formalized as an MDP, displayed using the `rich` package.

MDP definition:
    States S      : all 9 cells of a 3x3 grid, (2,2) is terminal (goal)
    Actions A     : UP, DOWN, LEFT, RIGHT
    Transitions P : deterministic -- moving off the grid bumps into a wall
                    and leaves the agent in place
    Rewards R     : +1.0 for entering the goal, -0.1 for every other step

Run:
    pip install rich
    python gridworld_mdp_rich.py
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.columns import Columns
from rich.text import Text

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


def build_mdp():
    """
    Returns mdp[(state, action)] = [(prob, next_state, reward, done)]
    Deterministic MDP: exactly one (prob=1.0) outcome per (state, action).
    """
    mdp = {}
    for s in STATES:
        if s == GOAL:
            continue  # terminal state -> no outgoing transitions
        r, c = s
        for a, (dr, dc) in ACTIONS.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                nr, nc = r, c  # wall bump: stay in place
            next_state = (nr, nc)
            reward = GOAL_REWARD if next_state == GOAL else STEP_REWARD
            done = next_state == GOAL
            mdp[(s, a)] = [(1.0, next_state, reward, done)]
    return mdp


MDP = build_mdp()

# --------------------------------- Display ---------------------------------- #

console = Console()


def state_label(s):
    return "GOAL" if s == GOAL else f"{s}"


def render_grid_overview():
    """Small visual reference grid, so the reader knows where each state sits."""
    table = Table(show_header=False, show_lines=True, padding=0, title="Grid layout")
    for _ in range(GRID_SIZE):
        table.add_column(justify="center", width=7)
    for r in range(GRID_SIZE):
        cells = []
        for c in range(GRID_SIZE):
            s = (r, c)
            if s == GOAL:
                cells.append(Text(f" {s}\nGOAL", style="bold black on yellow", justify="center"))
            elif s == (0, 0):
                cells.append(Text(f" {s}\nSTART", style="bold white on blue", justify="center"))
            else:
                cells.append(Text(f" {s} ", style="dim", justify="center"))
        table.add_row(*cells)
    return table


def render_full_transition_table():
    """One row per (state, action, next_state, prob, reward)."""
    table = Table(title="GridWorld MDP: Transition & Reward Table", show_lines=False)
    table.add_column("State s", justify="center", style="cyan")
    table.add_column("Action a", justify="center", style="magenta")
    table.add_column("Next state s'", justify="center", style="cyan")
    table.add_column("P(s'|s,a)", justify="center")
    table.add_column("R(s,a,s')", justify="center")

    for s in STATES:
        if s == GOAL:
            continue
        for a in ACTIONS:
            for prob, next_state, reward, done in MDP[(s, a)]:
                reward_style = "bold green" if reward == GOAL_REWARD else "red"
                table.add_row(
                    state_label(s),
                    a,
                    state_label(next_state),
                    f"{prob:.1f}",
                    Text(f"{reward:+.2f}", style=reward_style),
                )
    return table


def render_per_state_trees():
    """Group transitions by state, tree-style, for a more 'MDP dictionary' feel."""
    trees = []
    for s in STATES:
        if s == GOAL:
            root = Tree(f"[bold yellow]{state_label(s)}[/bold yellow] (terminal, no actions)")
            trees.append(root)
            continue
        root = Tree(f"[bold cyan]State {s}[/bold cyan]")
        for a in ACTIONS:
            prob, next_state, reward, done = MDP[(s, a)][0]
            reward_style = "bold green" if reward == GOAL_REWARD else "red"
            branch_text = (
                f"[magenta]{a}[/magenta] -> {state_label(next_state)}  "
                f"(P={prob:.1f}, R=[{reward_style}]{reward:+.2f}[/{reward_style}])"
            )
            root.add(branch_text)
        trees.append(root)
    return trees


def render_mdp_summary():
    summary = Table(show_header=False, box=None, padding=(0, 1))
    summary.add_row("States |S|:", f"{len(STATES)} (one terminal: {GOAL})")
    summary.add_row("Actions |A|:", f"{len(ACTIONS)} -> {', '.join(ACTIONS)}")
    summary.add_row("Transition type:", "Deterministic (P = 1.0 for the single outcome)")
    summary.add_row("Reward on goal entry:", f"{GOAL_REWARD:+.2f}")
    summary.add_row("Reward per other step:", f"{STEP_REWARD:+.2f}")
    summary.add_row("Discount factor γ:", "not fixed by env -- choose e.g. 0.9 / 0.99")
    return Panel(summary, title="MDP Summary", border_style="cyan")


def main():
    console.print(Panel.fit("GridWorld as a Formal MDP", style="bold magenta"))
    console.print(render_mdp_summary())
    console.print()
    console.print(render_grid_overview())
    console.print()
    console.print(render_full_transition_table())
    console.print()

    console.rule("[bold]Per-state view (dictionary style)[/bold]")
    for tree in render_per_state_trees():
        console.print(tree)
        console.print()


if __name__ == "__main__":
    main()
