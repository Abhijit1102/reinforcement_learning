# Chapter: Bellman Equations in Reinforcement Learning

> *"The value of where you are = the reward you get now + the value of where you end up."*

---

## Table of Contents

1. [The Core Idea](#1-the-core-idea)
2. [The Three Fundamental Equations](#2-the-three-fundamental-equations)
3. [Visual Intuition](#3-visual-intuition)
4. [Key Definitions](#4-key-definitions)
5. [From Equations to Algorithms](#5-from-equations-to-algorithms)
6. [The Big Picture](#6-the-big-picture)
7. [One-Sentence Takeaways](#7-one-sentence-takeaways)

---

## 1. The Core Idea

The Bellman equation is the **mathematical backbone** of Reinforcement Learning. It breaks the problem of finding optimal behavior into smaller, recursive subproblems.

Instead of trying to figure out the *entire* future at once, the Bellman equation says:

> **"The value of a state depends only on the immediate reward plus the value of the next state."**

This recursive property is what makes RL tractable — we can solve big problems by bootstrapping from smaller ones.

---

## 2. The Three Fundamental Equations

### 2.1 Bellman Expectation Equation for V<sup>π</sup>

Describes the value of following a **specific policy** π:

```
V^π(s) = Σₐ π(a|s) · Σₛ' P(s'|s,a) · [R(s,a,s') + γ·V^π(s')]
```

| Term | Meaning |
|------|---------|
| `π(a\|s)` | Probability of taking action `a` in state `s` under policy π |
| `P(s'\|s,a)` | Probability of transitioning to state `s'` |
| `R(s,a,s')` | Immediate reward received |
| `γ` | Discount factor (0 ≤ γ ≤ 1) |
| `V^π(s')` | Value of the next state (recursive!) |

---

### 2.2 Bellman Optimality Equation for V*

Describes the **best possible value** from any state:

```
V*(s) = maxₐ Σₛ' P(s'|s,a) · [R(s,a,s') + γ·V*(s')]
```

> Instead of averaging over actions (like the expectation equation), we **take the maximum** — because an optimal agent always picks the best action.

---

### 2.3 Bellman Optimality Equation for Q*

Describes the **best possible value** of taking a specific action:

```
Q*(s,a) = Σₛ' P(s'|s,a) · [R(s,a,s') + γ·maxₐ' Q*(s',a')]
```

> Once you know Q*, the optimal policy is trivial: **always pick the action with the highest Q-value.**

---

## 3. Visual Intuition

### 3.1 The Recursive Nature

```mermaid
graph TD
    S["🎯 State s"] -->|"take action a"| A["⚡ Action a"]
    A -->|"with prob P(s'|s,a)"| S1["📍 State s'₁"]
    A -->|"with prob P(s'|s,a)"| S2["📍 State s'₂"]
    A -->|"with prob P(s'|s,a)"| S3["📍 State s'₃"]

    S1 -->|"reward R₁"| V1["V(s'₁)"]
    S2 -->|"reward R₂"| V2["V(s'₂)"]
    S3 -->|"reward R₃"| V3["V(s'₃)"]

    V1 -.->|"γ · V(s'₁)"| S
    V2 -.->|"γ · V(s'₂)"| S
    V3 -.->|"γ · V(s'₃)"| S

    style S fill:#e1f5fe
    style A fill:#fff3e0
```

> **Key Insight:** The value of state `s` flows back from the values of all possible next states `s'`.

---

### 3.2 Policy Evaluation vs. Optimality

```mermaid
graph LR
    subgraph "Bellman Expectation<br/>V^π(s)"
        E1["State s"] -->|"π(a|s)"| EA["Average over<br/>all actions"]
        EA -->|"Σ"| EV["V^π(s)"]
    end

    subgraph "Bellman Optimality<br/>V*(s)"
        O1["State s"] -->|"maxₐ"| OA["Pick BEST<br/>action only"]
        OA -->|"max"| OV["V*(s)"]
    end

    style E1 fill:#e8f5e9
    style O1 fill:#ffebee
```

> **The only difference:** Expectation *averages* over actions (weighted by policy), while Optimality *maximizes* over actions.

---

### 3.3 V* vs Q* Relationship

```mermaid
graph TD
    subgraph "State s"
        S["V*(s)"]
    end

    subgraph "Actions available"
        A1["Q*(s,a₁)"]
        A2["Q*(s,a₂)"]
        A3["Q*(s,a₃)"]
        AD["..."]
    end

    S -->|"equals max of"| A1
    S --> A2
    S --> A3
    S --> AD

    A1 -->|"if chosen, leads to"| S1["V*(s'₁)"]
    A2 -->|"if chosen, leads to"| S2["V*(s'₂)"]
    A3 -->|"if chosen, leads to"| S3["V*(s'₃)"]

    style S fill:#e3f2fd
    style A1 fill:#fff8e1
    style A2 fill:#fff8e1
    style A3 fill:#fff8e1
```

> **V*(s) = maxₐ Q*(s,a)** — The best state value is simply the best action value available from that state.

---

### 3.4 The Optimal Policy from Q*

```mermaid
graph LR
    S["State s"] --> Q1["Q*(s,↑) = 5.2"]
    S --> Q2["Q*(s,→) = 8.7"]
    S --> Q3["Q*(s,↓) = 3.1"]
    S --> Q4["Q*(s,←) = 6.4"]

    Q1 -->|"not best"| X["❌"]
    Q2 -->|"highest!"| CHECK["✅ Optimal Action"]
    Q3 -->|"not best"| X
    Q4 -->|"not best"| X

    CHECK -->|"π*(s) = →"| NEXT["Go Right"]

    style Q2 fill:#c8e6c9
    style CHECK fill:#a5d6a7
```

> **Optimal Policy:** π*(s) = argmaxₐ Q*(s,a) — Always pick the action with the highest Q-value.

---

### 3.5 Full MDP Backup Diagram

```mermaid
graph TD
    subgraph "Backup from Q to V"
        QS["Q(s,a)"] -->|"weighted by π(a|s)"| VS["V(s)"]
    end

    subgraph "Backup from V to Q"
        QQ["Q(s,a)"] -->|"expectation over s' "| VN["R + γ·V(s')"]
    end

    subgraph "Optimal Backup"
        QO["Q*(s,a)"] -->|"max over a'"| VO["R + γ·max Q*(s',a')"]
    end

    style QS fill:#e3f2fd
    style VS fill:#e8f5e9
    style QQ fill:#fff3e0
    style QO fill:#ffebee
```

---

## 4. Key Definitions

| Symbol | Name | Meaning |
|--------|------|---------|
| **V<sup>π</sup>(s)** | State-Value Function | Expected return when starting from state `s` and following policy π |
| **V*(s)** | Optimal State-Value | Best possible expected return from state `s` |
| **Q<sup>π</sup>(s,a)** | Action-Value Function | Expected return when taking action `a` in state `s`, then following policy π |
| **Q*(s,a)** | Optimal Action-Value | Best possible expected return after taking action `a` in state `s` |
| **π*(s)** | Optimal Policy | The policy that achieves V*(s): π*(s) = argmaxₐ Q*(s,a) |

---

## 5. From Equations to Algorithms

```mermaid
flowchart TD
    A["Bellman<br/>Equation"] --> B{"What do we<br/>want to find?"}

    B -->|"Fixed policy π"| C["Policy<br/>Evaluation"]
    C -->|"Iterative update"| D["V^π(s) = Σ π(a|s)·Σ P(s'|s,a)·[R + γ·V^π(s')]"]

    B -->|"Best policy"| E["Value<br/>Iteration"]
    E -->|"Iterative update"| F["V*(s) = maxₐ Σ P(s'|s,a)·[R + γ·V*(s')]"]

    B -->|"Action values"| G["Q-Learning"]
    G -->|"Sample-based update"| H["Q*(s,a) = R + γ·maxₐ' Q*(s',a')"]

    D --> I["V^π converges"]
    F --> J["V* converges"]
    H --> K["Q* converges"]

    J -->|"π*(s) = argmaxₐ"| L["Optimal Policy"]
    K -->|"π*(s) = argmaxₐ"| L

    style A fill:#e3f2fd
    style L fill:#c8e6c9
```

---

## 6. The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    BELLMAN EQUATIONS                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   EXPECTATION          vs.           OPTIMALITY              │
│                                                             │
│   V^π(s) = E[R + γV^π(s')]        V*(s) = max E[R + γV*(s')]│
│          ↑                                ↑                 │
│   "Follow policy π"                "Be as smart as possible" │
│                                                             │
│   ┌─────────────┐                  ┌─────────────┐         │
│   │  POLICY     │                  │   VALUE     │         │
│   │ EVALUATION  │                  │  ITERATION  │         │
│   └─────────────┘                  └─────────────┘         │
│                                                             │
│   Q^π(s,a) = E[R + γV^π(s')]     Q*(s,a) = E[R + γmaxQ*]  │
│          ↑                                ↑                 │
│   "How good is this                "How good is this        │
│    action under π?"                action if we're         │
│                                      optimal after?"        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. One-Sentence Takeaways

- **V<sup>π</sup>** tells you how good it is to be in a state *if you keep following policy π*.
- **V*** tells you how good it is to be in a state *if you act perfectly from now on*.
- **Q*** tells you how good each *action* is — so you can simply pick the best one.
- The **recursive structure** (value depends on value) is what lets us solve RL problems iteratively.

---

*End of Chapter*
