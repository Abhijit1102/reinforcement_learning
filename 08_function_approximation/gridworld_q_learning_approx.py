"""
GridWorld — Q-Learning with Function Approximation
=====================================================

So far Q has been a TABLE: one number stored per (state, action) pair.
That only works because the grid is tiny (16 states x 4 actions = 64 numbers).
Real problems have huge or continuous state spaces where a table is
impossible to store. The fix is FUNCTION APPROXIMATION: represent
Q(s,a) as the output of a parametric function Q(s,a; theta) instead of a
lookup table, and update theta with gradient descent.

Semi-gradient Q-learning update (replaces the tabular update):

    target   = r + gamma * max_a' Q(s', a'; theta)
    error    = target - Q(s, a; theta)
    theta   <- theta + alpha * error * grad_theta[ Q(s, a; theta) ]

"Semi-gradient" because we don't differentiate through the target
(max_a' Q(s',a'; theta)) even though theta appears there too -- we treat
it as a fixed number, same as in tabular Q-learning. This is standard
practice and keeps training stable.

State representation: one-hot encoding of the (row, col) cell -> a length
16 vector with a single 1. This is the simplest possible feature vector;
swap in your own features (e.g. row, col, distance-to-goal) to see how
that changes learning.

Two approximators are implemented from scratch with numpy (no torch
needed for a network this small):

  1. LinearApproximator   Q(s,a) = w_a . x(s) + b_a           (one weight
                           vector per action -- this is the "linear"
                           option from the prompt)

  2. NeuralApproximator    Q(s,·) = W2 . ReLU(W1 . x(s) + b1) + b2
                           (a single hidden layer -- the "small neural
                           network" option from the prompt)

Flip APPROXIMATOR below to switch between them.
"""

import random
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
APPROXIMATOR = "nn"          # "linear" or "nn"
HIDDEN_SIZE = 32              # only used when APPROXIMATOR == "nn"

GRID_SIZE = 4
GAMMA = 1.0
ALPHA = 0.01 if APPROXIMATOR == "nn" else 0.1
EPSILON = 0.1
NUM_EPISODES = 2000
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
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = len(ACTIONS)

console = Console()
random.seed(42)
np.random.seed(42)


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


def one_hot(state):
    """Feature vector x(s): a 16-dim one-hot encoding of the grid cell."""
    x = np.zeros(N_STATES)
    r, c = state
    x[r * GRID_SIZE + c] = 1.0
    return x


# ----------------------------------------------------------------------------
# Approximator 1: Linear   Q(s,a) = w_a . x(s) + b_a
# ----------------------------------------------------------------------------
class LinearApproximator:
    def __init__(self, n_features, n_actions):
        self.W = np.zeros((n_features, n_actions))   # one weight column per action
        self.b = np.zeros(n_actions)

    def q_values(self, x):
        return x @ self.W + self.b                    # shape (n_actions,)

    def update(self, x, action_idx, td_error, alpha):
        # dQ(s,a)/dW[:,a] = x ,  dQ(s,a)/db[a] = 1
        self.W[:, action_idx] += alpha * td_error * x
        self.b[action_idx] += alpha * td_error


# ----------------------------------------------------------------------------
# Approximator 2: 1-hidden-layer NN   Q(s,·) = W2 . ReLU(W1.x + b1) + b2
# ----------------------------------------------------------------------------
class NeuralApproximator:
    def __init__(self, n_features, n_actions, hidden_size):
        # small random init keeps early Q-values near zero and stable
        self.W1 = np.random.randn(n_features, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, n_actions) * 0.1
        self.b2 = np.zeros(n_actions)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h = np.maximum(z1, 0.0)          # ReLU
        q = h @ self.W2 + self.b2
        return q, h                       # keep h around for the backward pass

    def q_values(self, x):
        q, _ = self.forward(x)
        return q

    def update(self, x, action_idx, td_error, alpha):
        q, h = self.forward(x)
        # gradient of 0.5*(target - q_a)^2 w.r.t. q_a is -(target - q_a) = -td_error
        # gradient descent step on that loss w.r.t. each parameter:
        d_q = np.zeros(N_ACTIONS)
        d_q[action_idx] = td_error         # ascend in the direction that reduces error

        # output layer
        self.W2 += alpha * np.outer(h, d_q)
        self.b2 += alpha * d_q

        # backprop into hidden layer (only through the column for action_idx)
        d_h = self.W2[:, action_idx] * td_error
        d_h[h <= 0] = 0.0                  # ReLU derivative

        self.W1 += alpha * np.outer(x, d_h)
        self.b1 += alpha * d_h


def make_approximator():
    if APPROXIMATOR == "linear":
        return LinearApproximator(N_STATES, N_ACTIONS)
    return NeuralApproximator(N_STATES, N_ACTIONS, HIDDEN_SIZE)


Q_approx = make_approximator()


def q_of(state):
    if state in TERMINAL_STATES:
        return np.zeros(N_ACTIONS)
    return Q_approx.q_values(one_hot(state))


def epsilon_greedy_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    q = q_of(state)
    best = np.max(q)
    best_actions = [ACTIONS[i] for i in range(N_ACTIONS) if q[i] == best]
    return random.choice(best_actions)


# ----------------------------------------------------------------------------
# Tracking for the convergence plot
# ----------------------------------------------------------------------------
TRACKED_STATES = [(1, 1), (2, 2), (0, 3), (3, 0)]
tracked_history = {s: [] for s in TRACKED_STATES}
episode_td_deltas = []


def train():
    for episode in range(1, NUM_EPISODES + 1):
        state = random.choice(non_terminal_states())
        total_abs_delta = 0.0
        steps = 0

        while state not in TERMINAL_STATES and steps < MAX_STEPS_PER_EPISODE:
            action = epsilon_greedy_action(state, EPSILON)
            action_idx = ACTIONS.index(action)
            s_next = next_state(state, action)
            moved = s_next != state
            r = reward(s_next, moved) if moved else -1

            x = one_hot(state)
            q_sa = Q_approx.q_values(x)[action_idx]
            target = r + GAMMA * np.max(q_of(s_next))
            td_error = target - q_sa
            td_error = np.clip(td_error, -20.0, 20.0)   # guard against early runaway updates

            Q_approx.update(x, action_idx, td_error, ALPHA)

            total_abs_delta += abs(td_error)
            state = s_next
            steps += 1

        episode_td_deltas.append(total_abs_delta / max(steps, 1))
        for s in TRACKED_STATES:
            tracked_history[s].append(float(np.max(q_of(s))))


# ----------------------------------------------------------------------------
# rich console output
# ----------------------------------------------------------------------------
def make_value_table():
    table = Table(title=f"V(s) = max_a Q(s,a; θ)  —  {APPROXIMATOR} approximator",
                  show_header=False, show_lines=True)
    for _ in range(GRID_SIZE):
        table.add_column(justify="center", width=9)
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            s = (r, c)
            row.append("[bold green]TERMINAL[/bold green]" if s in TERMINAL_STATES
                        else f"{np.max(q_of(s)):6.2f}")
        table.add_row(*row)
    return table


def make_policy_table():
    table = Table(title="Greedy Policy from Approximated Q", show_header=False, show_lines=True)
    for _ in range(GRID_SIZE):
        table.add_column(justify="center", width=9)
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            s = (r, c)
            if s in TERMINAL_STATES:
                row.append("[bold green]TERMINAL[/bold green]")
            else:
                q = q_of(s)
                best_a = ACTIONS[int(np.argmax(q))]
                row.append(f"[bold cyan]{ARROWS[best_a]}[/bold cyan]")
        table.add_row(*row)
    return table


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def make_convergence_plot(path=None):
    if path is None:
        import os
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"q_learning_{APPROXIMATOR}_convergence.png",
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    episodes_axis = list(range(1, NUM_EPISODES + 1))

    for s, history in tracked_history.items():
        ax1.plot(episodes_axis, history, label=f"V{s}")
    ax1.set_title(f"max_a Q(s,a; θ) converging  [{APPROXIMATOR}]")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Estimated value")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    window = 20
    rolling = [
        sum(episode_td_deltas[max(0, i - window):i + 1]) / len(episode_td_deltas[max(0, i - window):i + 1])
        for i in range(len(episode_td_deltas))
    ]
    ax2.plot(episodes_axis, episode_td_deltas, alpha=0.3, label="raw avg |TD error| per step")
    ax2.plot(episodes_axis, rolling, color="crimson", linewidth=2, label=f"{window}-episode rolling avg")
    ax2.set_title("TD-error magnitude shrinking as θ converges")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Avg |TD error| per step")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"Q-Learning with {APPROXIMATOR} function approximation  "
        f"(α={ALPHA}, γ={GAMMA}, ε={EPSILON}, {NUM_EPISODES} episodes)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    console.print(
        Panel(
            f"[bold]Q-Learning with Function Approximation[/bold]  ({APPROXIMATOR})\n"
            f"4x4 GridWorld | α = {ALPHA} | γ = {GAMMA} | ε = {EPSILON} | episodes = {NUM_EPISODES}\n"
            "Q(s,a;θ) replaces the table -- θ updated by semi-gradient Q-learning",
            border_style="blue",
        )
    )

    train()

    console.print(f"\n[bold green]Finished {NUM_EPISODES} episodes[/bold green]\n")
    console.print(make_value_table())
    console.print()
    console.print(make_policy_table())

    plot_path = make_convergence_plot()
    console.print(f"\n[bold]Convergence plot saved to:[/bold] {plot_path}")
