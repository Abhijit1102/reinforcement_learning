# A very simple way to demonstrate Reinforcement Learning (RL) without using any external packages is to create a small Grid World game.

# 🤖 Agent (A) tries to reach the Goal (G).
# ⭐ Reward = +10 for reaching the goal.
# ❌ Reward = -1 for every move.
# 🧠 The agent learns using the Q-Learning algorithm.
# 📦 Uses only Python's built-in libraries (random).

import random

SIZE = 5
ACTIONS = ["up", "down", "left", "right"]

# Q-table
Q = {}

def get_q(state, action):
    return Q.get((state, action), 0.0)

def choose_action(state, epsilon=0.2):
    if random.random() < epsilon:
        return random.choice(ACTIONS)

    values = [get_q(state, a) for a in ACTIONS]
    best = max(values)
    best_actions = [a for a, v in zip(ACTIONS, values) if v == best]
    return random.choice(best_actions)

def move(pos, action):
    x, y = pos

    if action == "up":
        x = max(0, x - 1)
    elif action == "down":
        x = min(SIZE - 1, x + 1)
    elif action == "left":
        y = max(0, y - 1)
    elif action == "right":
        y = min(SIZE - 1, y + 1)

    return (x, y)

alpha = 0.1
gamma = 0.9

goal = (4, 4)

# Training
for episode in range(500):

    state = (0, 0)

    while state != goal:

        action = choose_action(state)
        next_state = move(state, action)

        reward = 10 if next_state == goal else -1

        old_q = get_q(state, action)

        future = max(get_q(next_state, a) for a in ACTIONS)

        new_q = old_q + alpha * (reward + gamma * future - old_q)

        Q[(state, action)] = new_q

        state = next_state


# Test learned policy
state = (0, 0)

print("\nLearned Path\n")

while state != goal:

    print(state)

    action = choose_action(state, epsilon=0)

    state = move(state, action)

print(goal)