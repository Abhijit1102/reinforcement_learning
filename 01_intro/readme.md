# 🧠 Reinforcement Learning from Scratch (No External Libraries)

A beginner-friendly implementation of **Reinforcement Learning (RL)** using pure Python. This project demonstrates how an agent learns to make decisions through trial and error using the **Q-Learning** algorithm without relying on external machine learning libraries.

---

# 📖 What is Reinforcement Learning?

Reinforcement Learning (RL) is a branch of Machine Learning in which an **agent** learns how to make decisions by interacting with an **environment**.

Instead of learning from labeled data (Supervised Learning) or discovering patterns (Unsupervised Learning), an RL agent learns by receiving **rewards** or **penalties** for its actions.

The objective is to maximize the **total cumulative reward** over time.

---

# Real-Life Analogy

Imagine teaching a dog to sit.

* The dog performs an action.
* If it sits, you give it a treat.
* If it doesn't, it receives no reward.
* After many attempts, the dog learns that sitting results in a reward.

Reinforcement Learning follows the same principle.

```
Action
   ↓
Environment
   ↓
Reward
   ↓
Learning
   ↓
Better Action Next Time
```

---

# Core Components of Reinforcement Learning

## 1. Agent

The learner or decision-maker.

Example:

* Robot
* Self-driving car
* Game character

---

## 2. Environment

Everything the agent interacts with.

Examples:

* Chess board
* Video game
* Robot world
* Grid world

---

## 3. State (S)

A snapshot of the environment at a particular moment.

Example:

```
A . . .
. . . .
. . . G
```

The agent's current position represents the state.

---

## 4. Action (A)

A decision the agent can make.

Example:

```
Move Up
Move Down
Move Left
Move Right
```

---

## 5. Reward (R)

Feedback from the environment.

Example:

| Event       | Reward |
| ----------- | ------ |
| Reach Goal  | +10    |
| Normal Move | -1     |
| Hit Wall    | -5     |

Rewards encourage desirable behavior.

---

## 6. Policy (π)

A strategy that tells the agent which action to take in each state.

Initially:

```
Random Decisions
```

After learning:

```
Best Possible Decisions
```

---

## 7. Episode

One complete run from the starting state until the goal (or termination).

Example:

```
Start
 ↓
Move
 ↓
Move
 ↓
Move
 ↓
Goal
```

One complete journey is called an **episode**.

---

# Reinforcement Learning Process

```
        +-----------------------+
        |     Environment       |
        +-----------+-----------+
                    |
                Current State
                    |
                    v
        +-----------------------+
        |        Agent          |
        +-----------+-----------+
                    |
                 Action
                    |
                    v
        +-----------------------+
        |     Environment       |
        +-----------+-----------+
                    |
             Reward + Next State
                    |
                    +-------------> Repeat
```

The cycle continues until the task is completed.

---

# Exploration vs Exploitation

One of the biggest challenges in RL is balancing:

## Exploration

Trying new actions to discover better strategies.

Example:

```
Maybe going left is faster?
```

---

## Exploitation

Using the best strategy already learned.

Example:

```
I already know going right is best.
```

Most RL algorithms combine both approaches.

---

# What is Q-Learning?

Q-Learning is one of the simplest and most popular Reinforcement Learning algorithms.

It learns the **quality (Q-value)** of taking an action in a given state.

Higher Q-values indicate better actions.

Example:

| State | Action | Q-value |
| ----- | ------ | ------- |
| (0,0) | Right  | 5.6     |
| (0,0) | Left   | -2.1    |
| (0,0) | Up     | 3.8     |

The agent eventually chooses the action with the highest Q-value.

---

# Q-Table

A Q-table stores learned values for every state-action pair.

Example:

| State |  Up | Down | Left | Right |
| ----- | --: | ---: | ---: | ----: |
| (0,0) | 0.0 |  0.2 | -1.0 |   4.5 |
| (0,1) | 1.2 |  2.4 |  0.5 |   5.3 |

As training progresses, these values improve.

---

# Q-Learning Update Equation

```
Q(s,a) = Q(s,a)
         + α × [R + γ × max(Q(next_state)) − Q(s,a)]
```

Where:

* **Q(s,a)** → Current Q-value
* **α (Alpha)** → Learning rate
* **γ (Gamma)** → Discount factor
* **R** → Immediate reward
* **max(Q(next_state))** → Best estimated future reward

The equation updates the agent's knowledge after every action.

---

# Learning Workflow

```
Start Episode
      │
      ▼
Observe Current State
      │
      ▼
Choose an Action
      │
      ▼
Perform the Action
      │
      ▼
Receive Reward
      │
      ▼
Observe New State
      │
      ▼
Update Q-value
      │
      ▼
Goal Reached?
      │
 ┌────┴────┐
 │         │
No        Yes
 │         │
 └──Repeat─┘
```

---

# Grid World Example

```
+---+---+---+---+---+
| A |   |   |   |   |
+---+---+---+---+---+
|   |   |   |   |   |
+---+---+---+---+---+
|   |   |   |   |   |
+---+---+---+---+---+
|   |   |   |   |   |
+---+---+---+---+---+
|   |   |   |   | G |
+---+---+---+---+---+
```

* **A** = Agent (Start)
* **G** = Goal
* Empty cells = Free space

The objective is to discover the shortest path while maximizing cumulative reward.

---

# Advantages

* Learns from interaction instead of labeled data.
* Can solve sequential decision-making problems.
* Adapts to changing environments.
* Useful for robotics, games, finance, and autonomous systems.

---

# Limitations

* Requires many interactions to learn.
* Exploration can be time-consuming.
* Large state spaces require significant memory.
* Hyperparameter tuning can be challenging.

---

# Applications

* 🎮 Game Playing
* 🚗 Self-Driving Cars
* 🤖 Robotics
* 📈 Stock Trading
* 📦 Warehouse Automation
* 💬 Recommendation Systems
* 🌐 Resource Allocation
* ⚡ Energy Management

---

# Project Structure

```
reinforcement-learning/
│
├── main.py          # Q-Learning implementation
├── README.md        # Project documentation
└── LICENSE
```

---

# References

* Richard S. Sutton & Andrew G. Barto, *Reinforcement Learning: An Introduction*
* Q-Learning by Christopher Watkins (1989)
* OpenAI Spinning Up (RL Educational Resources)

---

# License

This project is provided for educational purposes. Feel free to modify, experiment with, and build upon it.
