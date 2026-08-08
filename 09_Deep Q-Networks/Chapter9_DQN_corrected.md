# Chapter 9: Deep Q-Networks (DQN) — RL Meets Deep Learning

> **Prerequisite**: You should understand Q-Learning (Chapter 8) before reading this chapter.

---

## 📌 Table of Contents

1. [Core Concepts](#core-concepts)
2. [DQN Architecture](#dqn-architecture)
3. [Experience Replay](#experience-replay)
4. [Target Network](#target-network)
5. [Loss Function & Bellman Update](#loss-function--bellman-update)
6. [The DQN Algorithm](#the-dqn-algorithm)
7. [Simple Analogies](#simple-analogies)
8. [Key Hyperparameters](#key-hyperparameters)
9. [Further Reading](#further-reading)

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **DQN** | Uses a deep neural network to approximate `Q(s,a)` instead of a Q-table. |
| **Experience Replay** | Stores past transitions `(s, a, r, s', done)` in a buffer. Training samples random batches to break correlation between consecutive experiences. |
| **Target Network** | A separate "frozen" network used to compute Q-targets. Copied from the main Q-network every `C` steps. Stabilizes training. |
| **Loss Function** | `L = (r + γ·max Q_target(s',a') − Q(s,a))²` |

---

## DQN Architecture

```mermaid
flowchart LR
    subgraph Input
        S[State s<br/>pixels/vector]
    end

    subgraph "Q-Network (Online)"
        L1[Conv/FC Layers]
        L2[Hidden Layers]
        L3[Hidden Layers]
    end

    subgraph Output
        Q1["Q(s,a₁)"]
        Q2["Q(s,a₂)"]
        Qn["Q(s,aₙ)"]
    end

    S --> L1 --> L2 --> L3
    L3 --> Q1
    L3 --> Q2
    L3 --> Qn

    style S fill:#6366f1,color:#fff
    style L1 fill:#1e293b,color:#fff
    style L2 fill:#1e293b,color:#fff
    style L3 fill:#1e293b,color:#fff
    style Q1 fill:#10b981,color:#fff
    style Q2 fill:#10b981,color:#fff
    style Qn fill:#10b981,color:#fff
```

**How it works:**
1. Feed state `s` into the neural network
2. Network outputs Q-values for **all** actions
3. Pick action with highest Q-value (exploit) or random action (explore)

---

## Experience Replay

```mermaid
flowchart TD
    A[Agent interacts<br/>with Environment] --> B["Generate transition<br/>(s, a, r, s', done)"]
    B --> C{Replay Buffer<br/>Capacity: N}
    C -->|Store| D[Buffer Slot 1]
    C -->|Store| E[Buffer Slot 2]
    C -->|Store| F[Buffer Slot 3]
    C -->|...| G[Buffer Slot N]

    H[Training Step] --> I{Sample Random<br/>Mini-Batch}
    I -->|Pick| D
    I -->|Pick| F
    I -->|Pick| G

    I --> J[Compute Loss &<br/>Update Q-Network]

    style A fill:#3b82f6,color:#fff
    style C fill:#f59e0b,color:#fff
    style H fill:#22c55e,color:#fff
    style J fill:#ec4899,color:#fff
```

### Why Experience Replay?

| Problem | Solution |
|---------|----------|
| Consecutive samples are highly correlated | Random sampling breaks correlation |
| Each experience used only once | Reuse experiences multiple times |
| Non-stationary data distribution | Averaged distribution over many states |

---

## Target Network

```mermaid
sequenceDiagram
    participant Q as Q-Network (Online)
    participant T as Q-Target (Frozen)
    participant E as Environment
    participant B as Replay Buffer

    loop Every Training Step
        Q->>E: Select action a = ε-greedy(Q(s))
        E->>B: Store (s, a, r, s', done)
        B->>Q: Sample random batch
        Q->>T: Compute y = r + γ·max Q_target(s',a')
        T-->>Q: Return target values
        Q->>Q: Update θ via SGD on (y - Q(s,a))²
    end

    Note over Q,T: Every C steps (e.g., 1000)
    Q->>T: θ⁻ ← θ (copy weights)
    Note over T: Target network "freezes"<br/>until next sync
```

### Why Target Network?

Without a target network:
- `Q(s,a)` chases its own moving shadow
- Creates a **moving target problem** → instability & divergence

With a target network:
- `Q_target` provides **stable bootstrap targets**
- Reduces oscillations during training
- "Dog chasing its own tail" → "Student learning from a stable teacher"

---

## Loss Function & Bellman Update

```mermaid
flowchart LR
    subgraph "Target Computation"
        R[r]
        G[γ]
        QT["max Q_target(s',a')"]
        Y["y = r + γ·max Q_target(s',a')<br/>(or y = r if done)"]
    end

    subgraph "Prediction"
        QS["Q(s,a)"]
    end

    subgraph "Loss"
        L["L = (y - Q(s,a))²"]
    end

    R --> Y
    G --> Y
    QT --> Y
    Y --> L
    QS --> L

    style R fill:#ef4444,color:#fff
    style G fill:#8b5cf6,color:#fff
    style QT fill:#f59e0b,color:#fff
    style Y fill:#1e293b,color:#fff
    style QS fill:#3b82f6,color:#fff
    style L fill:#ec4899,color:#fff
```

**Bellman Equation for DQN:**
```
y = r + γ · max Q_target(s', a')      if not done
y = r                                   if done

Loss = MSE(y, Q_network(s, a))
```

---

## The DQN Algorithm

```mermaid
flowchart TD
    Start([Start]) --> Init[Initialize Q_network θ<br/>Initialize Q_target θ⁻ = θ<br/>Initialize Replay Buffer D<br/>Initialize global_step = 0]
    Init --> Episode[For each episode]
    Episode --> Reset[s ← env.reset]
    Reset --> Step[For each step t]
    Step --> Action["a ← ε-greedy(Q(s))<br/>with probability ε: random<br/>else: argmax Q(s,a)"]
    Action --> Env["s', r, done ← env.step(a)"]
    Env --> Store["D.store(s, a, r, s', done)"]
    Store --> Sample["Sample mini-batch<br/>from D"]
    Sample --> Compute["For each transition:<br/>y = r + γ·max Q_target(s',a')<br/>or y = r if done"]
    Compute --> Update["Update θ to minimize<br/>Σ(y - Q(s,a))²"]
    Update --> Incr["global_step += 1"]
    Incr --> Sync{"global_step % C == 0?"}
    Sync -->|Yes| Copy["θ⁻ ← θ<br/>(copy weights)"]
    Sync -->|No| CheckDone{done?}
    Copy --> CheckDone
    CheckDone -->|No| Step
    CheckDone -->|Yes| DecayEps["Decay epsilon<br/>(once per episode)"]
    DecayEps --> Episode
    Episode --> End([End])

    style Start fill:#22c55e,color:#fff
    style End fill:#ef4444,color:#fff
    style Copy fill:#f59e0b,color:#fff
    style Update fill:#3b82f6,color:#fff
```

### Pseudocode

```python
# 1. Initialize
Q_network = NeuralNetwork()      # Main network with weights θ
Q_target = copy(Q_network)       # Target network with weights θ⁻ = θ
replay_buffer = ReplayBuffer(capacity=N)

global_step = 0                  # persists across episodes — used for target sync

# 2. Training Loop
for episode in range(M):
    state = env.reset()
    for t in range(T):
        # ε-greedy action selection
        if random() < epsilon:
            action = env.action_space.sample()    # Explore
        else:
            action = argmax(Q_network(state))      # Exploit

        next_state, reward, done = env.step(action)

        # Store experience
        replay_buffer.store(state, action, reward, next_state, done)

        # Sample and train
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)

            targets = []
            for s, a, r, s_next, done_flag in batch:
                if done_flag:
                    y = r
                else:
                    y = r + gamma * max(Q_target(s_next))
                targets.append(y)

            # Gradient descent on MSE loss
            loss = MSE(targets, Q_network(batch.states, batch.actions))
            Q_network.update(loss)

        # Sync target network every C GLOBAL steps (not per-episode step count!)
        global_step += 1
        if global_step % C == 0:
            Q_target.copy_weights_from(Q_network)

        state = next_state
        if done:
            break

    # Decay epsilon ONCE PER EPISODE (matches the ε_decay definition in the
    # hyperparameter table below: "Exploration decay per episode")
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

> ⚠️ **Common implementation pitfalls**
> 1. **Target sync counter reset bug** — if you use the per-episode loop variable `t` for `t % C == 0`, the sync fires near the start of *every* episode (since `t` restarts at 0 each episode) instead of every `C` steps overall. Always use a persistent `global_step` counter.
> 2. **Epsilon decay placement** — decaying `epsilon` inside the inner step loop decays it every environment step, not every episode. That contradicts the "decay per episode" definition used throughout this chapter and produces a much faster, unintended exploration schedule. Decay it once, after the episode's step loop ends.

---

## Simple Analogies

> **Experience Replay** = *"Instead of learning only from what just happened, I keep a diary of all my experiences and randomly re-read pages."*

> **Target Network** = *"I have a 'stable teacher' who doesn't change their mind every second. I compare my guesses to the teacher's answers."*

---

## Key Hyperparameters

| Parameter | Symbol | Typical Value | Purpose |
|-----------|--------|---------------|---------|
| Discount Factor | γ | 0.95 - 0.99 | Weight of future rewards |
| Epsilon (start) | ε | 1.0 | Initial exploration rate |
| Epsilon (min) | ε_min | 0.01 | Minimum exploration rate |
| Epsilon decay | ε_decay | 0.995 | Exploration decay per episode |
| Replay Buffer Size | N | 10,000 - 1,000,000 | Max stored transitions |
| Batch Size | B | 32 - 64 | Samples per training step |
| Target Sync Frequency | C | 500 - 10,000 | Steps between target network updates |
| Learning Rate | α | 0.0001 - 0.001 | SGD step size |

---

## Further Reading

- **Original DQN Paper**: Mnih et al. (2015) — "Human-level control through deep reinforcement learning" (Nature)
- **Double DQN**: Van Hasselt et al. (2016) — Decouples action selection from evaluation
- **Dueling DQN**: Wang et al. (2016) — Separates value and advantage streams
- **Prioritized Experience Replay**: Schaul et al. (2016) — Samples important transitions more frequently

---

*End of Chapter 9*
