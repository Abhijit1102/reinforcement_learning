# Chapter 2: Markov Decision Processes (MDPs)

> **MDPs** are the mathematical backbone of Reinforcement Learning. They provide a clean, formal way to describe how an agent interacts with an environment — defining states, actions, transitions, rewards, and how we value the future.

---

## 🎯 The Markov Property

> *"The future depends only on the present, not the past."*

If you know the **current state**, you don't need the full history to predict what happens next. The past is irrelevant once the present is known.

```mermaid
flowchart LR
    subgraph "Without Markov Property ❌"
        direction TB
        H1[s₀] --> H2[s₁] --> H3[s₂] --> H4[s₃]
        H4 -->|Needs full history| A1[Predict s₄]
    end

    subgraph "With Markov Property ✅"
        direction TB
        M1[s₀] --> M2[s₁] --> M3[s₂] --> M4[s₃]
        M4 -->|Only needs s₃| A2[Predict s₄]
    end

    style H1 fill:#ef4444,color:#fff
    style H2 fill:#ef4444,color:#fff
    style H3 fill:#ef4444,color:#fff
    style H4 fill:#ef4444,color:#fff
    style M4 fill:#10b981,color:#fff
    style A2 fill:#10b981,color:#fff
```

**Mathematically:**

$$P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0)$$

---

## 🧩 The MDP Tuple: (S, A, P, R, γ)

An MDP is fully defined by five components:

```mermaid
flowchart TB
    subgraph "MDP = (S, A, P, R, γ)"
        direction TB

        S["🗺️ S — States<br/>All possible situations<br/>e.g., positions on a board"]
        A["🎮 A — Actions<br/>All possible moves<br/>e.g., up, down, left, right"]
        P["🎲 P(s'|s,a) — Transition Prob.<br/>Chance of landing in s' from s via a"]
        R["💰 R(s,a,s') — Reward<br/>Immediate payoff for a transition"]
        G["⏳ γ — Discount Factor<br/>0 ≤ γ ≤ 1, how much we value the future"]

        S --> A
        A --> P
        P --> R
        R --> G
    end

    style S fill:#4f46e5,color:#fff
    style A fill:#10b981,color:#fff
    style P fill:#f59e0b,color:#fff
    style R fill:#ef4444,color:#fff
    style G fill:#8b5cf6,color:#fff
```

| Symbol | Name | What It Means |
|--------|------|---------------|
| **S** | State Space | The set of all possible states the environment can be in |
| **A** | Action Space | The set of all possible actions the agent can take |
| **P(s'\|s,a)** | Transition Probability | Probability of ending up in state $s'$ after taking action $a$ in state $s$ |
| **R(s,a,s')** | Reward Function | The immediate reward received for transitioning from $s$ to $s'$ via action $a$ |
| **γ** | Discount Factor | A number between 0 and 1 that determines how much we care about future vs. immediate rewards |

---

## 🔄 The MDP Transition Cycle

At every timestep, the MDP follows this exact pattern:

```mermaid
sequenceDiagram
    participant Agent as 🤖 Agent
    participant Env as 🌍 Environment (MDP)

    Agent->>Env: I am in state sₜ
    Env->>Agent: Available actions A(sₜ)
    Agent->>Env: I choose action aₜ
    Env->>Agent: New state sₜ₊₁ (sampled from P)
    Env->>Agent: Reward rₜ = R(sₜ, aₜ, sₜ₊₁)
    Note over Agent,Env: Repeat...
```

```mermaid
flowchart LR
    S[sₜ] -->|Choose aₜ| A
    A -->|P(sₜ₊₁|sₜ,aₜ)| S2[sₜ₊₁]
    S2 -->|R(sₜ,aₜ,sₜ₊₁)| R[rₜ]
    S2 --> S

    style S fill:#4f46e5,color:#fff
    style A fill:#10b981,color:#fff
    style S2 fill:#4f46e5,color:#fff
    style R fill:#ef4444,color:#fff
```

---

## 🎲 Transition Probabilities in Action

Not all actions have guaranteed outcomes. The environment can be stochastic:

```mermaid
flowchart TD
    S["State s<br/>(On square 5)"] -->|Action: Roll 3| A

    A["🎲 Chance Node"] -->|P = 0.8| S1["s' = Square 8<br/>💰 Reward: +$100"]
    A -->|P = 0.15| S2["s' = Square 7<br/>💰 Reward: +$50"]
    A -->|P = 0.05| S3["s' = Square 2<br/>💰 Reward: −$20"]

    style S fill:#4f46e5,color:#fff
    style A fill:#f59e0b,color:#fff
    style S1 fill:#10b981,color:#fff
    style S2 fill:#3b82f6,color:#fff
    style S3 fill:#ef4444,color:#fff
```

> **Key insight:** Even with the same action, different outcomes are possible. The agent must learn to handle uncertainty.

---

## 🎲 Simple Analogy: MDP as a Board Game Rulebook

> MDP is like a board game rulebook. It tells you: *"If you're on square 5 (state) and roll a 3 (action), you'll land on square 8 (next state) and collect $100 (reward)."*

```mermaid
flowchart LR
    subgraph "Board Game Analogy"
        direction TB

        Square5["🏠 Square 5<br/>(State s)"]
        Roll["🎲 Roll a 3<br/>(Action a)"]
        Square8["🏁 Square 8<br/>(Next State s')"]
        Money["💵 Collect $100<br/>(Reward r)"]

        Square5 --> Roll
        Roll --> Square8
        Square8 --> Money
    end

    style Square5 fill:#4f46e5,color:#fff
    style Roll fill:#f59e0b,color:#fff
    style Square8 fill:#10b981,color:#fff
    style Money fill:#ef4444,color:#fff
```

---

## ⏳ Why γ (Gamma) Matters

The discount factor $γ$ controls how much the agent values future rewards compared to immediate ones.

```mermaid
graph LR
    subgraph "Reward Timeline"
        direction LR

        Now["t = 0<br/>Reward: r₀"]
        T1["t = 1<br/>Reward: γ¹ · r₁"]
        T2["t = 2<br/>Reward: γ² · r₂"]
        T3["t = 3<br/>Reward: γ³ · r₃"]
        Dots["..."]
        Tn["t = n<br/>Reward: γⁿ · rₙ"]

        Now --> T1 --> T2 --> T3 --> Dots --> Tn
    end

    style Now fill:#ef4444,color:#fff
    style T1 fill:#f97316,color:#fff
    style T2 fill:#f59e0b,color:#fff
    style T3 fill:#eab308,color:#fff
    style Tn fill:#a3a3a3,color:#fff
```

### The Three Cases

| γ Value | Behavior | Analogy |
|---------|----------|---------|
| **γ = 0** | Agent is greedy. Only cares about **now**. | "A bird in the hand is worth two in the bush." |
| **γ = 1** | Agent cares about **all future rewards equally**. No discounting. | "A dollar today is worth exactly a dollar in 10 years." |
| **γ = 0.99** | **Standard choice.** Future matters, but immediate rewards still count more. | "Inflation exists — $100 today is worth more than $100 next year." |

```mermaid
graph TD
    subgraph "Discount Factor Spectrum"
        direction LR

        G0["γ = 0<br/>🍔 Myopic<br/>Only NOW matters"]
        G5["γ = 0.5<br/>😐 Balanced<br/>Short-term focused"]
        G99["γ = 0.99<br/>🔭 Farsighted<br/>Standard for RL"]
        G1["γ = 1<br/>♾️ Undiscounted<br/>All future equal"]

        G0 --> G5 --> G99 --> G1
    end

    style G0 fill:#ef4444,color:#fff
    style G5 fill:#f59e0b,color:#fff
    style G99 fill:#10b981,color:#fff
    style G1 fill:#3b82f6,color:#fff
```

> **The discount factor $γ$ is like inflation** — $100 today is worth more than $100 next year.

---

## 🏗️ The Return: Total Discounted Reward

The agent's goal is to maximize the **expected return** — the sum of all discounted future rewards:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

```mermaid
flowchart TB
    subgraph "Return Gₜ"
        direction LR

        R0["rₜ<br/>× 1"]
        R1["rₜ₊₁<br/>× γ"]
        R2["rₜ₊₂<br/>× γ²"]
        R3["rₜ₊₃<br/>× γ³"]
        Dots["..."]
        Rn["rₜ₊ₙ<br/>× γⁿ"]

        R0 -->|+| R1 -->|+| R2 -->|+| R3 -->|+| Dots -->|+| Rn
    end

    Sum["= Gₜ<br/>(Total Discounted Return)"]
    Rn --> Sum

    style R0 fill:#ef4444,color:#fff
    style R1 fill:#f97316,color:#fff
    style R2 fill:#f59e0b,color:#fff
    style R3 fill:#eab308,color:#fff
    style Rn fill:#a3a3a3,color:#fff
    style Sum fill:#10b981,color:#fff
```

---

## 📝 Key Takeaways

| Idea | One-Liner |
|------|-----------|
| **Markov Property** | The future depends only on the present state — history doesn't matter |
| **MDP Tuple** | (S, A, P, R, γ) fully describes any RL problem |
| **Stochastic Transitions** | Actions can have uncertain outcomes; P(s'\|s,a) captures this |
| **Discount Factor γ** | Controls the trade-off between immediate and future rewards |
| **Return Gₜ** | The sum of all discounted future rewards — what the agent tries to maximize |

---

## 🔗 Next Up

> **Chapter 3** — Value Functions: How good is a state? How good is an action in a state? Introducing $V(s)$ and $Q(s,a)$.

---

*Happy Learning! 🚀*
