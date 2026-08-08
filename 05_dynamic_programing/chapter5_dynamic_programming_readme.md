# Chapter 5: Dynamic Programming — Solving RL When You Know Everything

> *"If you know the rules of the game perfectly, you can compute the perfect strategy."*

---

## Table of Contents

1. [The Core Idea](#1-the-core-idea)
2. [Policy Iteration](#2-policy-iteration)
3. [Value Iteration](#3-value-iteration)
4. [Visual Intuition](#4-visual-intuition)
5. [Comparison](#5-comparison)
6. [Pseudocode](#6-pseudocode)
7. [Key Limitation](#7-key-limitation)
8. [One-Sentence Takeaways](#8-one-sentence-takeaways)

---

## 1. The Core Idea

Dynamic Programming (DP) is a collection of algorithms that **solve MDPs exactly** when the full model is known — i.e., you have access to all transition probabilities `P(s'|s,a)` and rewards `R(s,a,s')`.

DP exploits the recursive structure of the Bellman equations to iteratively improve solutions until convergence.

> **Two main approaches:**
> - **Policy Iteration** — Alternate between evaluating a policy and improving it.
> - **Value Iteration** — Skip explicit policies; directly compute optimal values, then extract the policy.

---

## 2. Policy Iteration

### The Loop

```
1. Start with any policy π₀
2. Policy Evaluation:    Compute V^π using Bellman expectation equation
3. Policy Improvement:   Update π to be greedy w.r.t. V^π
4. Repeat until π stops changing
```

### Policy Evaluation

Iteratively apply the Bellman expectation equation until V converges:

```
V_{k+1}(s) = Σₐ π(a|s) · Σₛ' P(s'|s,a) · [R(s,a,s') + γ · V_k(s')]
```

### Policy Improvement

For each state, pick the action that maximizes the expected return:

```
π'(s) = argmaxₐ Σₛ' P(s'|s,a) · [R(s,a,s') + γ · V^π(s')]
```

> **Guarantee:** Policy Improvement Theorem — the new policy π' is always at least as good as π, and strictly better unless π is already optimal.

---

## 3. Value Iteration

### The Idea

Skip the explicit policy. Directly compute V* by repeatedly applying the Bellman optimality equation:

```
V_{k+1}(s) = maxₐ Σₛ' P(s'|s,a) · [R(s,a,s') + γ · V_k(s')]
```

Once V* converges, extract the optimal policy in one step:

```
π*(s) = argmaxₐ Σₛ' P(s'|s,a) · [R(s,a,s') + γ · V*(s')]
```

> **Key Insight:** Value Iteration is essentially Policy Iteration where policy evaluation is truncated to **just one sweep** (one backup per state) before improving the policy.

---

## 4. Visual Intuition

### 4.1 Policy Iteration Loop

```mermaid
graph TD
    START(["Start"]) --> INIT["Initialize random policy π₀"]
    INIT --> EVAL["🔍 Policy Evaluation<br/>Compute V^π"]
    EVAL --> IMPROVE["⚡ Policy Improvement<br/>π' = greedy(V^π)"]
    IMPROVE --> CHECK{"π' == π?"}
    CHECK -->|"No"| UPDATE["π ← π'"]
    UPDATE --> EVAL
    CHECK -->|"Yes"| DONE(["✅ Optimal Policy Found"])

    style EVAL fill:#e3f2fd
    style IMPROVE fill:#fff3e0
    style DONE fill:#c8e6c9
```

---

### 4.2 Value Iteration Loop

```mermaid
graph TD
    START(["Start"]) --> INIT["Initialize V₀(s) = 0 for all s"]
    INIT --> UPDATE["🔄 Bellman Optimality Backup<br/>V_{k+1}(s) = maxₐ E[R + γ·V_k(s')]"]
    UPDATE --> CHECK{"Converged?"}
    CHECK -->|"No"| UPDATE
    CHECK -->|"Yes"| EXTRACT["🎯 Extract Policy<br/>π*(s) = argmaxₐ Q*(s,a)"]
    EXTRACT --> DONE(["✅ Optimal Policy Found"])

    style UPDATE fill:#e8f5e9
    style EXTRACT fill:#fff8e1
    style DONE fill:#c8e6c9
```

---

### 4.3 Policy Iteration vs Value Iteration Side-by-Side

```mermaid
graph TB
    subgraph "Policy Iteration"
        PI1["Policy Evaluation<br/>(many sweeps)"] --> PI2["Policy Improvement<br/>(one sweep)"]
        PI2 --> PI1
    end

    subgraph "Value Iteration"
        VI1["Bellman Optimality Backup<br/>(one sweep)"] --> VI2["Implicit Policy Improvement<br/>(built into max)"]
        VI2 --> VI1
    end

    PI1 -.->|"slower per iteration"| PI_LABEL["Fewer iterations<br/>but each is expensive"]
    VI1 -.->|"faster per iteration"| VI_LABEL["More iterations<br/>but each is cheap"]

    style PI1 fill:#e3f2fd
    style VI1 fill:#e8f5e9
```

---

### 4.4 The Policy Improvement Step

```mermaid
graph LR
    subgraph "Before Improvement"
        S1["State s₁<br/>π(s₁) = ↑<br/>V = 3.2"]
        S2["State s₂<br/>π(s₂) = →<br/>V = 4.1"]
        S3["State s₃<br/>π(s₃) = ↓<br/>V = 2.8"]
    end

    ARROW["Policy<br/>Improvement"] --> AFTER

    subgraph "After Improvement"
        A1["State s₁<br/>π'(s₁) = →<br/>V = 5.7 ✓"]
        A2["State s₂<br/>π'(s₂) = →<br/>V = 4.1"]
        A3["State s₃<br/>π'(s₃) = ←<br/>V = 4.2 ✓"]
    end

    style A1 fill:#c8e6c9
    style A3 fill:#c8e6c9
```

> Arrows (✓) mark states where the policy changed because a better action was found.

---

### 4.5 Value Iteration Convergence

```mermaid
graph LR
    subgraph "Iteration 0"
        V0["V₀(s) = 0<br/>∀s"]
    end

    subgraph "Iteration 1"
        V1["V₁(s) = maxₐ R(s,a)<br/>Immediate rewards only"]
    end

    subgraph "Iteration 2"
        V2["V₂(s) = maxₐ E[R + γ·V₁(s')]"]
    end

    subgraph "Iteration k"
        VK["V_k(s) = maxₐ E[R + γ·V_{k-1}(s')]"]
    end

    subgraph "Converged"
        VSTAR["V*(s)<br/>Optimal values!"]
    end

    V0 --> V1 --> V2 -->|"..."| VK -->|"k → ∞"| VSTAR

    style V0 fill:#ffebee
    style VSTAR fill:#c8e6c9
```

> Each iteration expands the "planning horizon" by one step. Eventually, the values converge to the true optimal values.

---

### 4.6 The Full DP Family Tree

```mermaid
graph TD
    DP["Dynamic Programming<br/>Know the full MDP"] --> PI["Policy Iteration"]
    DP --> VI["Value Iteration"]

    PI --> PE["Policy Evaluation<br/>Iterative V^π updates"]
    PI --> PIM["Policy Improvement<br/>Greedy action selection"]

    VI --> BO["Bellman Optimality<br/>One-step backups"]

    PE --> VPI["V^π converges"]
    PIM --> BETTER["π' ≥ π"]
    BO --> VSTAR["V* converges"]

    VPI --> PIM
    BETTER --> PE
    VSTAR --> EXTRACT["Extract π*"]

    style DP fill:#e3f2fd
    style VSTAR fill:#c8e6c9
    style EXTRACT fill:#c8e6c9
```

---

## 5. Comparison

| Aspect | Policy Iteration | Value Iteration |
|--------|-----------------|-----------------|
| **Core Loop** | Evaluate → Improve → Repeat | Backup with max → Repeat |
| **Policy stored?** | Yes, explicitly | No, implicit in V |
| **Convergence speed** | Fewer iterations | More iterations |
| **Cost per iteration** | Expensive (many sweeps) | Cheap (one sweep) |
| **When to use** | Good initial policy known | No good initial guess |
| **Guarantee** | Converges to π* | Converges to V* and π* |

---

## 6. Pseudocode

### Policy Iteration

```python
# Initialize
pi = random_policy()

while True:
    # Policy Evaluation
    V = evaluate_policy(pi, P, R, gamma)

    # Policy Improvement
    pi_new = improve_policy(V, P, R, gamma)

    if pi_new == pi:
        break
    pi = pi_new

return pi, V
```

### Value Iteration

```python
# Initialize
V = {s: 0 for s in states}

while not converged:
    delta = 0
    for s in states:
        v = V[s]
        V[s] = max_a sum_sprime(P(s'|s,a) * (R(s,a,s') + gamma * V[s']))
        delta = max(delta, abs(v - V[s]))
    if delta < theta:
        break

# Extract optimal policy
pi = {s: argmax_a Q(s,a) for s in states}
return pi, V
```

---

## 7. Key Limitation

```
┌─────────────────────────────────────────────────────────────┐
│                     ⚠️  THE CATCH                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Dynamic Programming requires:                            │
│                                                             │
│   ✅ Full knowledge of P(s'|s,a)  — Transition model       │
│   ✅ Full knowledge of R(s,a,s')  — Reward function         │
│                                                             │
│   ❌ This is RARELY true in real-world RL!                  │
│                                                             │
│   Example: A robot doesn't know the physics of every        │
│   surface. A game AI doesn't have the source code.          │
│                                                             │
│   → That's why we need Monte Carlo and Temporal Difference  │
│     methods (next chapters).                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. One-Sentence Takeaways

- **Policy Iteration** alternates between "how good is this policy?" (evaluation) and "can I make it better?" (improvement) until nothing changes.
- **Value Iteration** skips the policy — just keep applying the Bellman optimality equation until the values stop changing, then extract the policy.
- Both are guaranteed to converge to the **optimal policy** for finite MDPs.
- DP is powerful but **requires perfect knowledge** of the environment — a luxury we rarely have.
- When the model is unknown, we must **sample** from the environment instead — leading to Monte Carlo and TD methods.

---

## Analogy Recap

> **Policy Iteration** = "I have a map. I test my route (policy), measure how long it takes (evaluation), then find shortcuts (improvement), and repeat."
>
> **Value Iteration** = "I work backwards from the destination. I figure out the best time-to-goal from every intersection, then my route is obvious."

---

*End of Chapter 5*
