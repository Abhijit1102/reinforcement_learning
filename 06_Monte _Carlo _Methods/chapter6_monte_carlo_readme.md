# Chapter 6: Monte Carlo Methods — Learning from Experience

> *"Play the whole game, then learn from the final score."*

---

## Table of Contents

1. [The Core Idea](#1-the-core-idea)
2. [How Monte Carlo Works](#2-how-monte-carlo-works)
3. [First-Visit vs Every-Visit MC](#3-first-visit-vs-every-visit-mc)
4. [The MC Update Rule](#4-the-mc-update-rule)
5. [Visual Intuition](#5-visual-intuition)
6. [Pseudocode](#6-pseudocode)
7. [Pros & Cons](#7-pros--cons)
8. [One-Sentence Takeaways](#8-one-sentence-takeaways)

---

## 1. The Core Idea

**Monte Carlo (MC) methods** solve the reinforcement learning problem by **learning directly from experience** — no model of the environment required.

Instead of knowing transition probabilities `P(s'|s,a)`, the agent simply **plays full episodes** (from start to finish), observes what happens, and updates its value estimates based on the **actual returns** received.

> **Key Principle:** The agent doesn't plan ahead using a model. It tries things out, watches the outcome, and learns from the result.

This makes MC **model-free** — a huge advantage over Dynamic Programming, which required perfect knowledge of the MDP.

---

## 2. How Monte Carlo Works

### The Loop

```
1. Generate an episode using current policy π
   (e.g., S₀, A₀, R₁, S₁, A₁, R₂, ..., S_T)

2. For each state-action pair (s,a) visited in the episode:
   - Calculate the return G = sum of discounted rewards from that point onward
   - Update Q(s,a) using G

3. Improve the policy to be greedy w.r.t. Q

4. Repeat
```

> **Important:** MC only updates values **after the episode ends**. You must wait until the game is over before learning anything.

---

## 3. First-Visit vs Every-Visit MC

### First-Visit MC

For each state `s` (or state-action pair), only the **first occurrence** in an episode is used to update the estimate.

```
For episode e:
    For each state s in e:
        If s is visited for the FIRST time in e:
            G = return from first visit to end
            Append G to Returns(s)
    V(s) = average(Returns(s))
```

> **Why?** Later visits to the same state within the same episode are not independent — they depend on the first visit. First-visit gives unbiased estimates.

---

### Every-Visit MC

Every single visit to a state `s` in an episode contributes to the estimate.

```
For episode e:
    For each visit to state s in e:
        G = return from this visit to end
        Append G to Returns(s)
    V(s) = average(Returns(s))
```

> **Trade-off:** Every-visit has lower variance per episode but slightly more bias. In practice, both converge to the same answer.

---

## 4. The MC Update Rule

### For State Values

```
V(s) ← V(s) + α · [G - V(s)]
```

### For Action Values (Q)

```
Q(s,a) ← Q(s,a) + α · [G - Q(s,a)]
```

Where:

| Symbol | Meaning |
|--------|---------|
| `α` | Learning rate (0 < α ≤ 1). With α = 1/N, this becomes the sample average. |
| `G` | Actual return: `G = R_{t+1} + γ·R_{t+2} + γ²·R_{t+3} + ...` |
| `G - Q(s,a)` | **TD Error** — how wrong was our prediction? |

> **Intuition:** If the actual return `G` was higher than expected `Q(s,a)`, increase the estimate. If lower, decrease it.

---

## 5. Visual Intuition

### 5.1 The MC Episode Loop

```mermaid
graph TD
    START(["Start"]) --> POLICY["Current Policy π"]
    POLICY --> EPISODE["🎮 Play Full Episode<br/>S₀→A₀→R₁→S₁→A₁→R₂→...→S_T"]
    EPISODE --> WAIT["⏳ Wait until episode ends"]
    WAIT --> CALC["📊 Calculate Returns G<br/>for each (s,a) visited"]
    CALC --> UPDATE["🔄 Update Q(s,a)<br/>Q ← Q + α·[G - Q]"]
    UPDATE --> IMPROVE["⚡ Policy Improvement<br/>π ← ε-greedy(Q)"]
    IMPROVE --> MORE{"More episodes?"}
    MORE -->|"Yes"| EPISODE
    MORE -->|"No"| DONE(["✅ Learned Policy"])

    style EPISODE fill:#e3f2fd
    style WAIT fill:#ffebee
    style UPDATE fill:#fff3e0
    style DONE fill:#c8e6c9
```

> **Notice the WAIT step** — MC cannot learn mid-episode. Everything happens after the game ends.

---

### 5.2 First-Visit vs Every-Visit

```mermaid
graph LR
    subgraph "Episode Path"
        S1["S₁"] --> A1["A₁"]
        A1 --> S2["S₂"]
        S2 --> A2["A₂"]
        A2 --> S3["S₃"]
        S3 --> A3["A₃"]
        A3 --> S1_AGAIN["S₁ again!"]
        S1_AGAIN --> A4["A₄"]
        A4 --> S4["S₄ (terminal)"]
    end

    subgraph "First-Visit MC"
        FV["Only count<br/>FIRST visit to S₁"]
        FV --> G1["G from first S₁"]
    end

    subgraph "Every-Visit MC"
        EV["Count BOTH<br/>visits to S₁"]
        EV --> G1
        EV --> G2["G from second S₁"]
    end

    style S1 fill:#e3f2fd
    style S1_AGAIN fill:#ffebee
    style FV fill:#e8f5e9
    style EV fill:#fff3e0
```

> First-visit ignores the second occurrence of S₁. Every-visit counts both.

---

### 5.3 Return Calculation

```mermaid
graph LR
    subgraph "Episode"
        T0["t=0<br/>S=s, A=a"]
        T1["t=1<br/>R=+2"]
        T2["t=2<br/>R=+3"]
        T3["t=3<br/>R=-1"]
        T4["t=4<br/>R=+5<br/>Terminal"]
    end

    T0 --> T1 --> T2 --> T3 --> T4

    G["G(s,a) = 2 + γ·3 + γ²·(-1) + γ³·5"] --> FORMULA["= Σ γᵏ·R_{t+k+1}"]

    style T0 fill:#e3f2fd
    style G fill:#fff8e1
```

> The return G is the **discounted sum of all future rewards** from the point of visiting (s,a).

---

### 5.4 MC vs DP — The Model-Free Advantage

```mermaid
graph TB
    subgraph "Dynamic Programming"
        DP1["Needs P(s'|s,a)"] --> DP2["Plans using model"]
        DP2 --> DP3["Updates every state"]
        DP3 --> DP4["✅ Fast, exact<br/>❌ Needs perfect model"]
    end

    subgraph "Monte Carlo"
        MC1["No model needed"] --> MC2["Samples episodes"]
        MC2 --> MC3["Only updates visited states"]
        MC3 --> MC4["✅ Model-free<br/>❌ Must wait for episode end"]
    end

    style DP1 fill:#ffebee
    style MC1 fill:#c8e6c9
```

---

### 5.5 Exploration vs Exploitation in MC

```mermaid
graph TD
    Q["Q(s,a) estimates"] --> DECIDE{"Choose action?"}

    DECIDE -->|"With prob ε"| EXPLORE["🎲 Random action<br/>Explore new paths"]
    DECIDE -->|"With prob 1-ε"| EXPLOIT["⭐ Best known action<br/>argmax_a Q(s,a)"]

    EXPLORE --> EPISODE["Generate episode"]
    EXPLOIT --> EPISODE

    EPISODE --> UPDATE["Update Q values"]
    UPDATE --> Q

    style EXPLORE fill:#fff3e0
    style EXPLOIT fill:#e8f5e9
```

> MC uses **ε-greedy exploration** to ensure all state-action pairs are visited enough times.

---

### 5.6 The Learning Process Over Episodes

```mermaid
graph LR
    subgraph "Episode 1"
        E1["Q(s,a) = 0<br/>G = 12<br/>→ Q = 12"]
    end

    subgraph "Episode 2"
        E2["Q(s,a) = 12<br/>G = 8<br/>→ Q = 10"]
    end

    subgraph "Episode 3"
        E3["Q(s,a) = 10<br/>G = 15<br/>→ Q = 11.7"]
    end

    subgraph "Episode N"
        EN["Q(s,a) → true<br/>expected return"]
    end

    E1 --> E2 --> E3 -->|"..."| EN

    style E1 fill:#ffebee
    style EN fill:#c8e6c9
```

> Over many episodes, the Q-estimates converge to the true expected returns.

---

## 6. Pseudocode

### Monte Carlo Control (ε-greedy)

```python
# Initialize
Q = defaultdict(lambda: 0)      # Action-value function
Returns = defaultdict(list)      # Store returns for each (s,a)
pi = epsilon_greedy_policy(Q)   # Behavior policy

for episode in range(num_episodes):
    # Generate episode using pi
    episode_data = []
    s = env.reset()
    while not done:
        a = pi(s)                           # Sample action
        s_next, r, done = env.step(a)
        episode_data.append((s, a, r))
        s = s_next

    # Calculate returns and update Q
    G = 0
    visited = set()
    for t in reversed(range(len(episode_data))):
        s, a, r = episode_data[t]
        G = gamma * G + r                    # Discounted return

        # First-visit check
        if (s, a) not in visited:
            visited.add((s, a))
            Returns[(s, a)].append(G)
            Q[(s, a)] = np.mean(Returns[(s, a)])  # Or incremental update

    # Policy improvement (ε-greedy)
    pi = make_epsilon_greedy(Q, epsilon)

return Q, pi
```

---

## 7. Pros & Cons

```
┌─────────────────────────────────────────────────────────────┐
│                   MONTE CARLO: PROS & CONS                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✅ MODEL-FREE                                              │
│     No need to know P(s'|s,a) or R(s,a,s').                 │
│     Just interact with the environment and observe.         │
│                                                             │
│  ✅ UNBIASED ESTIMATES                                      │
│     Uses actual returns G, not bootstrapped estimates.      │
│     Converges to true values with enough episodes.          │
│                                                             │
│  ✅ WORKS FOR EPISODIC TASKS                                │
│     Great for games, puzzles, and any task with a clear     │
│     start and end.                                          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ❌ MUST WAIT FOR EPISODE END                               │
│     Can't learn mid-episode. If episodes are long or        │
│     infinite, MC doesn't work.                              │
│                                                             │
│  ❌ HIGH VARIANCE                                           │
│     One lucky/unlucky episode can swing estimates wildly.   │
│     Needs many episodes to stabilize.                       │
│                                                             │
│  ❌ ONLY UPDATES VISITED STATES                             │
│     States never visited get no updates. Exploration        │
│     strategy is critical.                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. One-Sentence Takeaways

- **Monte Carlo** learns by playing full episodes and updating estimates with the actual return — no model needed.
- **First-Visit MC** only counts the first time you see a state in an episode; **Every-Visit MC** counts all occurrences.
- The update rule `Q ← Q + α·[G - Q]` moves your estimate toward the actual observed return.
- MC is **model-free** and **unbiased**, but you must wait until the episode ends and variance can be high.
- If episodes are very long or never-ending, MC won't work — that's where Temporal Difference methods come in.

---

## Analogy Recap

> **Monte Carlo** is like reviewing a full basketball game tape. You watch the whole game (episode), then update your strategy based on whether you won or lost. You don't change your plan mid-game.

---

*End of Chapter 6*
