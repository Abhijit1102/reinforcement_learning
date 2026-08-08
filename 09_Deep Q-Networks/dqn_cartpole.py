"""
DQN Agent for CartPole-v1 (Gymnasium)
======================================
- Neural network: 2 hidden layers, 128 units each
- Experience replay buffer: capacity 10,000
- Target network, synced every N steps
- Rich console UI shows live progress: episode, reward, epsilon, loss, buffer size

Run:
    python dqn_cartpole.py
"""

import random
import time
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.layout import Layout
from rich import box

console = Console()

# ----------------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------------
ENV_NAME = "CartPole-v1"
HIDDEN_SIZE = 128
BUFFER_CAPACITY = 10_000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995        # multiplicative decay per episode
TARGET_UPDATE_EVERY = 10  # episodes
NUM_EPISODES = 300
SOLVE_SCORE = 475.0       # avg reward over last 100 episodes considered "solved"
MAX_STEPS_PER_EPISODE = 500
SEED = 42

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Q-Network: 2 hidden layers, 128 units each
# ----------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------------
# Experience Replay Buffer (capacity 10,000)
# ----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# ----------------------------------------------------------------------------
# DQN Agent
# ----------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = action_dim

        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

        self.epsilon = EPS_START
        self.steps_done = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def optimize(self):
        """One gradient step of DQN learning. Returns loss (float) or None if not enough data."""
        if len(self.replay_buffer) < BATCH_SIZE:
            return None

        batch = self.replay_buffer.sample(BATCH_SIZE)

        states = torch.as_tensor(np.array(batch.state), dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
        dones = torch.as_tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

        # Q(s,a) from policy network
        q_values = self.policy_net(states).gather(1, actions)

        # max_a' Q_target(s', a')  -- target network for stability
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + GAMMA * next_q_values * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)


# ----------------------------------------------------------------------------
# Rich UI helpers
# ----------------------------------------------------------------------------
def build_header_panel():
    return Panel(
        "[bold cyan]DQN Agent — CartPole-v1[/bold cyan]\n"
        f"[dim]Net: {HIDDEN_SIZE}-{HIDDEN_SIZE} hidden units | "
        f"Replay buffer: {BUFFER_CAPACITY} | Target sync every {TARGET_UPDATE_EVERY} episodes | "
        f"Device: {device}[/dim]",
        box=box.ROUNDED,
        style="cyan",
    )


def build_stats_table(episode, reward, avg_reward100, epsilon, loss, buffer_len, best_reward):
    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="bold green")

    table.add_row("Episode", f"{episode}/{NUM_EPISODES}")
    table.add_row("Reward (this episode)", f"{reward:.1f}")
    table.add_row("Avg reward (last 100)", f"{avg_reward100:.1f}")
    table.add_row("Best reward so far", f"{best_reward:.1f}")
    table.add_row("Epsilon (exploration)", f"{epsilon:.3f}")
    table.add_row("Last loss", f"{loss:.4f}" if loss is not None else "n/a (warming up buffer)")
    table.add_row("Replay buffer size", f"{buffer_len}/{BUFFER_CAPACITY}")
    return table


def build_layout(episode, reward, avg_reward100, epsilon, loss, buffer_len, best_reward, progress):
    layout = Layout()
    layout.split_column(
        Layout(build_header_panel(), size=4),
        Layout(Panel(build_stats_table(episode, reward, avg_reward100, epsilon, loss,
                                        buffer_len, best_reward),
                     title="Training Status", border_style="magenta"), size=12),
        Layout(Panel(progress, title="Progress", border_style="blue"), size=4),
    )
    return layout


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    reward_history = deque(maxlen=100)
    best_reward = -float("inf")
    last_loss = None

    progress = Progress(
        TextColumn("[bold blue]Training"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )
    task_id = progress.add_task("train", total=NUM_EPISODES)

    console.print(build_header_panel())
    console.print()

    with Live(console=console, refresh_per_second=8) as live:
        for episode in range(1, NUM_EPISODES + 1):
            state, _ = env.reset(seed=SEED + episode)
            episode_reward = 0.0

            for step in range(MAX_STEPS_PER_EPISODE):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store_transition(state, action, reward, next_state, float(done))
                state = next_state
                episode_reward += reward

                loss = agent.optimize()
                if loss is not None:
                    last_loss = loss

                agent.steps_done += 1

                live.update(build_layout(
                    episode, episode_reward,
                    np.mean(reward_history) if reward_history else 0.0,
                    agent.epsilon, last_loss, len(agent.replay_buffer), best_reward, progress
                ))

                if done:
                    break

            reward_history.append(episode_reward)
            best_reward = max(best_reward, episode_reward)
            agent.decay_epsilon()

            if episode % TARGET_UPDATE_EVERY == 0:
                agent.update_target_network()

            progress.update(task_id, advance=1)
            avg100 = np.mean(reward_history)

            live.update(build_layout(
                episode, episode_reward, avg100, agent.epsilon,
                last_loss, len(agent.replay_buffer), best_reward, progress
            ))

            if avg100 >= SOLVE_SCORE and len(reward_history) == 100:
                console.print(
                    f"\n[bold green]✅ Solved at episode {episode}! "
                    f"Avg reward over last 100 episodes: {avg100:.1f}[/bold green]"
                )
                break

    env.close()

    console.print()
    console.print(Panel(
        f"[bold]Training finished.[/bold]\n"
        f"Best single-episode reward: [green]{best_reward:.1f}[/green]\n"
        f"Final avg reward (last {len(reward_history)} episodes): "
        f"[green]{np.mean(reward_history):.1f}[/green]",
        title="Summary", border_style="green", box=box.ROUNDED,
    ))

    torch.save(agent.policy_net.state_dict(), "dqn_cartpole_policy.pt")
    console.print("[dim]Saved trained weights to dqn_cartpole_policy.pt[/dim]")


if __name__ == "__main__":
    train()
