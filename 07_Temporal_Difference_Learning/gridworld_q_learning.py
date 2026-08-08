"""
GridWorld — Q-Learning (off-policy TD control)
=================================================

Q-Learning update rule (Bellman optimality equation applied as a TD update,
using a SAMPLED transition instead of a full model):

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
                                 (--------- TD target ---------)  (----)
                                                                  old estimate

Key contrast with the earlier methods:
  - Value/Policy Iteration: needs a model of the environment, updates using
    the FULL expectation over all outcomes.
  - Monte Carlo:            model-free, but only updates at the END of an
                             episode using the actual observed return G_t.
  - Q-Learning:              model-free AND updates after EVERY single step
                             (bootstrapping off its own current estimate of
                             the next state's best action, rather than
                             waiting for the episode to finish). This makes
                             it a "TD(0)" method.
  - Off-policy:              the behavior policy is epsilon-greedy (it
                             explores), but the update always bootstraps
                             using max_a' Q(s',a') — the GREEDY action —
                             regardless of which action was actually taken
                             next. So it learns Q* even while exploring.
"""

import random
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# GridWorld setup (same world as previous mini-projects)
# ----------------------------------------------------------------------------
GRID_SIZE = 4
GAMMA = 1.0
ALPHA = 0.1                 # learning rate
EPSILON = 0.1                # exploration rate
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 200
TERMINAL_STATES = {(0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)}
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_DELTA = {
    "UP":    (-1, 0),
    "DOWN":  (1, 0),
    "LEFT":  (0, -1),
    "RIGHT": (0, 1),
}
ARROWS = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}

console = Console()
random.seed(42)


def all_states():
    return [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]


def non_terminal_states():
    return [s for s in all_states() if s not in TERMINAL_STATES]


def next_state(state, action):
    if state in TERMINAL_STATES:
        return state
    dr, dc = ACTION_DELTA[action]
    r, c = state
    nr, nc = r + dr, c + dc
    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
        return (nr, nc)
    return state


def reward(s_next, moved):
    if s_next in TERMINAL_STATES:
        return 0
    return -1


# ----------------------------------------------------------------------------
# Q-table
# ----------------------------------------------------------------------------
Q = {(s, a): 0.0 for s in all_states() for a in ACTIONS}


def epsilon_greedy_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    q_values = {a: Q[(state, a)] for a in ACTIONS}
    best = max(q_values.values())
    best_actions = [a for a, v in q_values.items() if v == best]
    return random.choice(best_actions)


def greedy_max_q(state):
    if state in TERMINAL_STATES:
        return 0.0
    return max(Q[(state, a)] for a in ACTIONS)


# ----------------------------------------------------------------------------
# Tracking for the convergence plot
# ----------------------------------------------------------------------------
# 1) A handful of representative (state, action) pairs, watched episode by episode
TRACKED_PAIRS = [
    ((1, 1), "LEFT"),
    ((2, 2), "RIGHT"),
    ((0, 3), "DOWN"),
    ((3, 0), "UP"),
]
tracked_history = {pair: [] for pair in TRACKED_PAIRS}

# 2) Overall per-episode TD-update magnitude, as a global convergence signal
episode_td_deltas = []


def train_q_learning():
    for episode in range(1, NUM_EPISODES + 1):
        state = random.choice(non_terminal_states())
        total_abs_delta = 0.0
        steps = 0

        while state not in TERMINAL_STATES and steps < MAX_STEPS_PER_EPISODE:
            action = epsilon_greedy_action(state, EPSILON)
            s_next = next_state(state, action)
            moved = s_next != state
            r = reward(s_next, moved) if moved else -1

            td_target = r + GAMMA * greedy_max_q(s_next)
            td_error = td_target - Q[(state, action)]
            Q[(state, action)] += ALPHA * td_error

            total_abs_delta += abs(td_error)
            state = s_next
            steps += 1

        episode_td_deltas.append(total_abs_delta / max(steps, 1))

        for (s, a) in TRACKED_PAIRS:
            tracked_history[(s, a)].append(Q[(s, a)])


# ----------------------------------------------------------------------------
# rich console output
# ----------------------------------------------------------------------------
def make_value_table():
    table = Table(title="V(s) = max_a Q(s,a)  —  final", show_header=False, show_lines=True)
    for _ in range(GRID_SIZE):
        table.add_column(justify="center", width=9)
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            s = (r, c)
            row.append("[bold green]TERMINAL[/bold green]" if s in TERMINAL_STATES
                        else f"{greedy_max_q(s):6.2f}")
        table.add_row(*row)
    return table


def make_policy_table():
    table = Table(title="Greedy Policy from Learned Q", show_header=False, show_lines=True)
    for _ in range(GRID_SIZE):
        table.add_column(justify="center", width=9)
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            s = (r, c)
            if s in TERMINAL_STATES:
                row.append("[bold green]TERMINAL[/bold green]")
            else:
                best_a = max(ACTIONS, key=lambda a: Q[(s, a)])
                row.append(f"[bold cyan]{ARROWS[best_a]}[/bold cyan]")
        table.add_row(*row)
    return table


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def make_convergence_plot(path=None):
    if path is None:
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_learning_convergence.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    episodes_axis = list(range(1, NUM_EPISODES + 1))
    for (s, a), history in tracked_history.items():
        ax1.plot(episodes_axis, history, label=f"Q{s}, {a}")
    ax1.set_title("Selected Q(s,a) values converging over episodes")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Q-value")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Rolling average of the TD-update magnitude shows the overall
    # convergence trend more clearly than the raw noisy signal.
    window = 20
    rolling = [
        sum(episode_td_deltas[max(0, i - window):i + 1]) / len(episode_td_deltas[max(0, i - window):i + 1])
        for i in range(len(episode_td_deltas))
    ]
    ax2.plot(episodes_axis, episode_td_deltas, alpha=0.3, label="raw avg |TD error| per step")
    ax2.plot(episodes_axis, rolling, color="crimson", linewidth=2, label=f"{window}-episode rolling avg")
    ax2.set_title("TD-error magnitude shrinking as Q converges")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Avg |TD error| per step")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Q-Learning convergence  (α={ALPHA}, γ={GAMMA}, ε={EPSILON}, {NUM_EPISODES} episodes)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    console.print(
        Panel(
            "[bold]Q-Learning (off-policy TD control)[/bold]\n"
            f"4x4 GridWorld | α = {ALPHA} | γ = {GAMMA} | ε = {EPSILON} | episodes = {NUM_EPISODES}\n"
            "Q(s,a) <- Q(s,a) + α [ r + γ·max_a' Q(s',a') - Q(s,a) ]",
            border_style="magenta",
        )
    )

    train_q_learning()

    console.print(f"\n[bold green]Finished {NUM_EPISODES} episodes[/bold green]\n")
    console.print(make_value_table())
    console.print()
    console.print(make_policy_table())

    plot_path = make_convergence_plot()
    console.print(f"\n[bold]Convergence plot saved to:[/bold] {plot_path}")
