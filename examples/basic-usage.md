# Basic Usage Examples

## Example 1: Automatic Optimization

```
You: "Refactor this function to be more efficient"

Plugin automatically:
1. Generates 3 variants (imperative, functional, hybrid)
2. Selects variant using Thompson Sampling
3. Executes and tracks performance
4. Updates Bayesian statistics
```

## Example 2: Explicit Variant Testing

```
You: "Test these API approaches:
- REST with Express
- GraphQL with Apollo
- tRPC with Zod"

Plugin creates experiment assignments and tracks which performs best
```

## Example 3: Check Results

```
You: "What's the best approach for API design based on past experiments?"

Plugin evaluates statistical significance and recommends winner
```

## Example 4: Evolve Prompts

```
You: "Evolve better prompts for database optimization tasks"

Plugin:
1. Selects top-performing variants
2. Applies genetic crossover
3. Introduces mutations
4. Creates new generation of prompts
```

## Example 5: Contextual Selection

```
You: "For frontend React components, which coding style works best?"

Plugin uses contextual bandits with features:
- Language: JavaScript/TypeScript
- Framework: React
- Task type: Component creation
- Complexity: Medium

Selects variant optimized for this specific context
```

## Example 6: Human Feedback

```
You: "This implementation is better than the previous one"

Plugin records preference comparison:
- Winner: Current variant
- Loser: Previous variant
- Context: Task description
- Updates Bradley-Terry ratings
```

## Example 7: Performance Monitoring

```
You: "Show me the performance stats for all variants"

Plugin displays:
- Success rate (Bayesian posterior mean)
- Average latency
- Token cost efficiency
- Confidence intervals
- Total trials
```
