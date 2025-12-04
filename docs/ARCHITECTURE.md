# Architecture Documentation

## Overview

The OpenCode A/B Learning Plugin implements a sophisticated self-learning system that combines multiple machine learning techniques for automated prompt optimization.

## Core Components

### 1. Thompson Sampling Engine

**Purpose**: Multi-armed bandit optimization for variant selection

**Implementation**:
- Maintains Beta posterior distributions for each variant
- Samples from Beta(α, β) using Gamma distribution composition
- Updates posteriors using Bayesian inference

**Mathematical Foundation**:
```
Posterior: Beta(α, β)
α = prior_α + successes
β = prior_β + failures

Sampling: X ~ Gamma(α, 1), Y ~ Gamma(β, 1)
Beta sample = X / (X + Y)
```

### 2. Genetic Algorithm Module

**Purpose**: Evolutionary optimization of prompt variants

**Components**:
- **Selection**: Tournament selection with k=3
- **Crossover**: LLM-based intelligent combination
- **Mutation**: Random perturbations with 20% probability
- **Fitness**: Average reward from Bayesian posterior

**Evolution Process**:
```
1. Select top N performers (elitism)
2. Tournament selection for parents
3. LLM-based crossover
4. Mutation with probability p_mut
5. Create offspring
6. Repeat for G generations
```

### 3. Contextual Bandit (LinUCB)

**Purpose**: Context-aware variant selection

**Algorithm**:
```
Score(variant, context) = 
  reward_estimate(variant, context) + 
  α × uncertainty(variant, context)

where:
  reward_estimate = θᵀx (linear model)
  uncertainty = √(xᵀA⁻¹x) (confidence radius)
```

**Context Features**:
- Task complexity (length, structure)
- Language/framework indicators
- Historical performance patterns
- Time-based features

### 4. RLHF Preference Learning

**Purpose**: Human-in-the-loop preference optimization

**Model**: Bradley-Terry pairwise preferences
```
P(variant_i > variant_j) = 
  exp(θᵢ) / (exp(θᵢ) + exp(θⱼ))
```

**Update Rule**:
- Winner: α ← α + λ (boost success count)
- Loser: β ← β + λ (boost failure count)
- λ = learning rate (default 0.1)

### 5. Database Layer

**Technology**: SQLite for persistence

**Schema Design**:
```sql
prompt_variants
  ├─ id (PK)
  ├─ template
  ├─ parent_id (FK)
  ├─ generation
  └─ created_at

variant_performance
  ├─ variant_id (FK)
  ├─ alpha (Bayesian)
  ├─ beta (Bayesian)
  ├─ total_trials
  ├─ avg_reward
  ├─ avg_latency_ms
  └─ avg_token_cost

experiment_assignments
  ├─ task_id (PK)
  ├─ variant_id (FK)
  ├─ context
  └─ timestamp

feedback_records
  ├─ id (PK)
  ├─ task_id
  ├─ variant_id (FK)
  ├─ reward
  ├─ latency_ms
  ├─ token_cost
  ├─ success
  └─ timestamp

preference_comparisons
  ├─ id (PK)
  ├─ winner_id (FK)
  ├─ loser_id (FK)
  ├─ context
  ├─ human_feedback
  └─ timestamp

contextual_features
  ├─ id (PK)
  ├─ variant_id (FK)
  ├─ context_hash
  ├─ features (BLOB)
  ├─ reward
  └─ timestamp
```

## Data Flow

### Standard Execution Flow

```
┌────────────────┐
│  User Query    │
└───────┬────────┘
        │
        ▼
┌────────────────────────┐
│  OpenCode Router       │
│  (Identifies Task)     │
└───────┬────────────────┘
        │
        ▼
┌────────────────────────┐
│  Plugin: Thompson      │
│  Sampling Selection    │◄──── Bayesian Posteriors
└───────┬────────────────┘      (alpha, beta)
        │
        ▼
┌────────────────────────┐
│  Execute with Variant  │
│  (LLM API Call)        │
└───────┬────────────────┘
        │
        ▼
┌────────────────────────┐
│  Collect Metrics       │
│  - Success/Failure     │
│  - Latency             │
│  - Token Cost          │
└───────┬────────────────┘
        │
        ▼
┌────────────────────────┐
│  Update Database       │
│  - Bayesian Update     │
│  - Performance Metrics │
└───────┬────────────────┘
        │
        ▼
┌────────────────────────┐
│  Return Result to User │
└────────────────────────┘
```

### Background Learning Loop

```
┌──────────────────┐
│  Hourly Trigger  │
└────────┬─────────┘
         │
         ▼
┌─────────────────────┐
│  Statistical Tests  │
│  (Variant Pruning)  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Genetic Algorithm  │
│  (Evolve Prompts)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Decay Exploration  │
│  Rate (ε ← 0.95ε)   │
└────────┬────────────┘
         │
         └──────┐
                │
                ▼
        ┌──────────────┐
        │ Sleep 1 hour │
        └──────┬───────┘
               │
               └─────► Loop
```

## Key Algorithms

### Thompson Sampling Selection

```typescript
function thompsonSample(candidateIds: string[]): string {
  const samples = candidateIds.map(id => {
    const { alpha, beta } = getPerformance(id);
    return sampleBeta(alpha, beta);
  });
  
  return candidateIds[argmax(samples)];
}
```

### Bayesian Update

```typescript
function recordFeedback(feedback: FeedbackRecord): void {
  const { alpha, beta, totalTrials } = getPerformance(feedback.variantId);
  
  const newAlpha = alpha + (feedback.success ? 1 : 0);
  const newBeta = beta + (feedback.success ? 0 : 1);
  const newTrials = totalTrials + 1;
  
  updateDatabase(feedback.variantId, {
    alpha: newAlpha,
    beta: newBeta,
    totalTrials: newTrials
  });
}
```

### Statistical Significance Test

```typescript
function evaluateABTest(
  variantA: string,
  variantB: string
): { winner: string | null; confidence: number } {
  const perfA = getPerformance(variantA);
  const perfB = getPerformance(variantB);
  
  // Monte Carlo simulation
  let countAWins = 0;
  for (let i = 0; i < 10000; i++) {
    const sampleA = sampleBeta(perfA.alpha, perfA.beta);
    const sampleB = sampleBeta(perfB.alpha, perfB.beta);
    if (sampleA > sampleB) countAWins++;
  }
  
  const probAWins = countAWins / 10000;
  const confidence = Math.max(probAWins, 1 - probAWins);
  
  return {
    winner: confidence > 0.95 ? 
      (probAWins > 0.5 ? variantA : variantB) : null,
    confidence
  };
}
```

## Performance Characteristics

### Time Complexity

- **Thompson Sampling**: O(k) where k = number of candidates
- **Bayesian Update**: O(1) per feedback
- **Statistical Test**: O(n × s) where n = simulations (10k), s = samples (2)
- **Genetic Algorithm**: O(g × p × c) where g = generations, p = population, c = crossover cost

### Space Complexity

- **Per Variant**: ~200 bytes (metadata + statistics)
- **Per Experiment**: ~150 bytes (assignment + feedback)
- **Database Growth**: ~1MB per 5000 experiments

### Latency

- **Selection Decision**: 10-50ms
- **Feedback Recording**: 5-20ms
- **Statistical Test**: 100-500ms
- **Evolution (async)**: 2-5 seconds

## Configuration Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| exploration_rate | 0.3 → 0.05 | [0.01, 0.5] | Initial exploration, decays over time |
| temperature | 1.0 | [0.1, 2.0] | Softmax temperature for action selection |
| min_trials | 10 | [5, 100] | Minimum trials before exploitation |
| evolution_interval | 3600s | [300, 86400] | Time between evolution cycles |
| population_size | 5 | [3, 20] | Number of variants in genetic pool |
| mutation_rate | 0.2 | [0.05, 0.5] | Probability of mutation |
| pruning_threshold | 0.05 | [0.01, 0.2] | Min win rate to avoid pruning |

## Extension Points

### Custom Reward Functions

Implement `RewardFunction` interface:
```typescript
interface RewardFunction {
  calculate(
    success: boolean,
    latencyMs: number,
    tokenCost: number,
    context: any
  ): number;
}
```

### Custom Context Extractors

Implement `ContextExtractor` interface:
```typescript
interface ContextExtractor {
  extract(task: string, metadata: any): number[];
}
```

### Custom Evolution Operators

Implement `EvolutionOperator` interface:
```typescript
interface EvolutionOperator {
  crossover(parentA: string, parentB: string): Promise<string>;
  mutate(prompt: string): Promise<string>;
}
```

## Testing Strategy

### Unit Tests
- Thompson sampling correctness
- Bayesian update accuracy
- Statistical test validity
- Database operations

### Integration Tests
- End-to-end experiment flow
- Background learning loop
- Multi-variant selection

### Performance Tests
- Selection latency under load
- Database scalability
- Memory usage over time

## Future Enhancements

1. **Neural Contextual Bandits**: Replace LinUCB with deep learning
2. **Multi-Objective Optimization**: NSGA-II for Pareto frontier
3. **Transfer Learning**: Knowledge sharing across projects
4. **Hierarchical Thompson Sampling**: Multi-level decision making
5. **Automated Hyperparameter Tuning**: Meta-optimization
