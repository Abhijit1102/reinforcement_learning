"""
Simple 3x3 GridWorld with a random agent, visualized live using the `rich` package.

Environment:
    - 3x3 grid, states are (row, col) with row/col in {0,1,2}
    - Start:  (0, 0)
    - Goal:   (2, 2)
    - Reward: +1.0 for reaching the goal, -0.1 for every other step
    - Actions: up, down, left, right (moves that would leave the grid are ignored,
      but still cost -0.1, just like bumping into a wall)

Run:
    pip install rich
    python gridworld_rich.py
"""

import random
import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ----------------------------- Environment ------------------------------- #

GRID_SIZE = 3
START = (0, 0)
GOAL = (2, 2)
STEP_REWARD = -0.1
GOAL_REWARD = 1.0
MAX_STEPS = 50

ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


class GridWorld:
    def __init__(self):
        self.state = START
        self.done = False
        self.step_count = 0
        self.total_reward = 0.0

    def reset(self):
        self.state = START
        self.done = False
        self.step_count = 0
        self.total_reward = 0.0
        return self.state

    def step(self, action: str):
        if self.done:
            raise RuntimeError("Episode already finished. Call reset().")

        dr, dc = ACTIONS[action]
        r, c = self.state
        new_r, new_c = r + dr, c + dc

        # If the move goes off the grid, the agent stays in place (wall bump).
        if 0 <= new_r < GRID_SIZE and 0 <= new_c < GRID_SIZE:
            self.state = (new_r, new_c)

        self.step_count += 1

        if self.state == GOAL:
            reward = GOAL_REWARD
            self.done = True
        else:
            reward = STEP_REWARD
            if self.step_count >= MAX_STEPS:
                self.done = True

        self.total_reward += reward
        return self.state, reward, self.done


# ------------------------------- Rendering -------------------------------- #

def render_grid(state) -> Table:
    table = Table(show_header=False, show_lines=True, expand=False, padding=0)
    for _ in range(GRID_SIZE):
        table.add_column(justify="center", width=5)

    for r in range(GRID_SIZE):
        row_cells = []
        for c in range(GRID_SIZE):
            if (r, c) == state == GOAL:
                cell = Text(" A/G ", style="bold white on green")
            elif (r, c) == state:
                cell = Text("  A  ", style="bold white on blue")
            elif (r, c) == GOAL:
                cell = Text("  G  ", style="bold black on yellow")
            else:
                cell = Text("  .  ", style="dim")
            row_cells.append(cell)
        table.add_row(*row_cells)
    return table


def render_layout(env: GridWorld, action: str, reward: float, episode: int) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(
        Layout(name="grid", ratio=1),
        Layout(name="stats", ratio=1),
    )

    layout["header"].update(
        Panel(
            Text(f"GridWorld  |  Episode {episode}  |  Step {env.step_count}", justify="center"),
            style="bold cyan",
        )
    )

    layout["grid"].update(Panel(render_grid(env.state), title="Grid (A=agent, G=goal)"))

    stats_table = Table(show_header=False, box=None, padding=(0, 1))
    stats_table.add_row("Action taken:", action if action else "-")
    stats_table.add_row("Agent position:", str(env.state))
    stats_table.add_row("Step reward:", f"{reward:+.2f}")
    stats_table.add_row("Cumulative reward:", f"{env.total_reward:+.2f}")
    stats_table.add_row("Steps so far:", str(env.step_count))
    stats_table.add_row("Done:", str(env.done))

    layout["stats"].update(Panel(stats_table, title="Parameters / State"))

    return layout


# --------------------------------- Main ------------------------------------ #

def run_episode(env: GridWorld, live: Live, episode: int, delay: float = 0.5):
    env.reset()
    live.update(render_layout(env, action="-", reward=0.0, episode=episode))
    time.sleep(delay)

    while not env.done:
        action = random.choice(list(ACTIONS.keys()))
        _, reward, done = env.step(action)
        live.update(render_layout(env, action=action, reward=reward, episode=episode))
        time.sleep(delay)

    return env.total_reward, env.step_count


def main():
    console = Console()
    console.print(Panel.fit("Random-Agent GridWorld Demo", style="bold magenta"))

    num_episodes = 3
    delay_seconds = 0.4  # lower = faster animation

    env = GridWorld()

    with Live(console=console, refresh_per_second=10) as live:
        for episode in range(1, num_episodes + 1):
            total_reward, steps = run_episode(env, live, episode, delay=delay_seconds)
            time.sleep(0.8)

    console.print("\n[bold green]Done![/bold green] Ran", num_episodes, "episode(s).")


if __name__ == "__main__":
    main()