"""
GridWorld — First-Visit Monte Carlo Control (epsilon-greedy)
==============================================================

Unlike Policy/Value Iteration, Monte Carlo methods don't need a model of
the environment (no p(s',r|s,a) table). Instead they learn Q(s,a) purely
from sampled episodes:

    1. Generate an episode by following an epsilon-greedy policy derived
       from the current Q-table.
    2. Compute the return G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...
    3. For the FIRST time each (s,a) pair appears in the episode, add G_t
       to that pair's running average:

           Q(s,a) ← average of all first-visit returns seen so far for (s,a)

    4. Improve the policy to be epsilon-greedy w.r.t. the updated Q.

Repeated over many episodes, Q(s,a) converges to Q*(s,a), and the greedy
policy w.r.t. Q converges to the optimal policy π*.
"""

import random
import time
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

# ----------------------------------------------------------------------------
# GridWorld setup (same world as previous mini-projects)
# ----------------------------------------------------------------------------
GRID_SIZE = 4
GAMMA = 1.0
EPSILON = 0.1
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 200          # safety cap against runaway episodes
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


def reward(state):
    return 0 if state in TERMINAL_STATES else -1


# ----------------------------------------------------------------------------
# Q-table and epsilon-greedy policy
# ----------------------------------------------------------------------------
Q = {(s, a): 0.0 for s in all_states() for a in ACTIONS}
returns_sum = defaultdict(float)
returns_count = defaultdict(int)


def epsilon_greedy_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    q_values = {a: Q[(state, a)] for a in ACTIONS}
    best = max(q_values.values())
    best_actions = [a for a, v in q_values.items() if v == best]
    return random.choice(best_actions)  # break ties randomly


def generate_episode(epsilon):
    """
    Exploring start: begin from a random non-terminal state so every
    state gets visited over the course of training, then follow the
    epsilon-greedy policy until a terminal state or step cap.
    """
    episode = []
    state = random.choice(non_terminal_states())
    for _ in range(MAX_STEPS_PER_EPISODE):
        action = epsilon_greedy_action(state, epsilon)
        s_next = next_state(state, action)
        r = reward(s_next) if s_next != state else -1
        episode.append((state, action, r))
        state = s_next
        if state in TERMINAL_STATES:
            break
    return episode


def first_visit_mc_update(episode):
    """Apply the first-visit MC update rule to Q using this episode's returns."""
    G = 0.0
    visited_pairs = set()
    # Walk backwards to accumulate discounted return G_t at each step
    returns_at_t = [0.0] * len(episode)
    for t in reversed(range(len(episode))):
        _, _, r = episode[t]
        G = r + GAMMA * G
        returns_at_t[t] = G

    for t, (s, a, _) in enumerate(episode):
        if (s, a) in visited_pairs:
            continue  # only the FIRST occurrence counts
        visited_pairs.add((s, a))
        returns_sum[(s, a)] += returns_at_t[t]
        returns_count[(s, a)] += 1
        Q[(s, a)] = returns_sum[(s, a)] / returns_count[(s, a)]


# ----------------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------------
def state_value_from_Q(state):
    if state in TERMINAL_STATES:
        return 0.0
    return max(Q[(state, a)] for a in ACTIONS)


def make_value_table(episode_num):
    table = Table(
        title=f"V(s) = max_a Q(s,a)  —  after episode {episode_num}/{NUM_EPISODES}",
        show_header=False,
        show_lines=True,
    )
    for _ in range(GRID_SIZE):
        table.add_column(justify="center", width=9)
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            s = (r, c)
            if s in TERMINAL_STATES:
                row.append("[bold green]TERMINAL[/bold green]")
            else:
                row.append(f"{state_value_from_Q(s):6.2f}")
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


def make_q_table():
    """Detailed Q(s,a) table, one row per state, one column per action."""
    table = Table(title="Learned Q(s, a) table", show_lines=True)
    table.add_column("State", justify="center", style="bold")
    for a in ACTIONS:
        table.add_column(a, justify="center")
    for s in all_states():
        if s in TERMINAL_STATES:
            table.add_row(str(s), *(["[green]—[/green]"] * len(ACTIONS)))
        else:
            best_val = max(Q[(s, a)] for a in ACTIONS)
            cells = []
            for a in ACTIONS:
                val = Q[(s, a)]
                text = f"{val:6.2f}"
                if val == best_val:
                    text = f"[bold yellow]{text}[/bold yellow]"
                cells.append(text)
            table.add_row(str(s), *cells)
    return table


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train(animate=True, refresh_every=20):
    if animate:
        with Live(make_value_table(0), console=console, refresh_per_second=8) as live:
            for ep in range(1, NUM_EPISODES + 1):
                episode = generate_episode(EPSILON)
                first_visit_mc_update(episode)
                if ep % refresh_every == 0 or ep == NUM_EPISODES:
                    live.update(make_value_table(ep))
    else:
        for ep in range(1, NUM_EPISODES + 1):
            episode = generate_episode(EPSILON)
            first_visit_mc_update(episode)


if __name__ == "__main__":
    console.print(
        Panel(
            "[bold]First-Visit Monte Carlo Control (epsilon-greedy)[/bold]\n"
            f"4x4 GridWorld | γ = {GAMMA} | ε = {EPSILON} | episodes = {NUM_EPISODES}\n"
            "Q(s,a) ← average of first-visit returns G_t across all sampled episodes",
            border_style="yellow",
        )
    )

    start = time.time()
    train(animate=True, refresh_every=20)
    elapsed = time.time() - start

    console.print(f"\n[bold green]Finished {NUM_EPISODES} episodes in {elapsed:.1f}s[/bold green]\n")
    console.print(make_value_table(NUM_EPISODES))
    console.print()
    console.print(make_policy_table())
    console.print()
    console.print(make_q_table())
