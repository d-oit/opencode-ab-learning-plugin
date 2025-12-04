# Advanced Usage Examples

## Custom Reward Functions

```typescript
// Define custom reward that balances multiple objectives
const customReward = (
  success: boolean,
  latencyMs: number,
  tokenCost: number
) => {
  const successWeight = 10.0;
  const latencyPenalty = latencyMs / 1000.0;
  const costPenalty = tokenCost * 0.001;
  
  return success ? 
    (successWeight - latencyPenalty - costPenalty) : 
    -5.0;
};
```

## Contextual Feature Engineering

```typescript
// Extract rich contextual features for LinUCB
function extractContextFeatures(task: string): number[] {
  return [
    task.length / 1000,                    // Normalized length
    (task.match(/function/g) || []).length, // Function count
    (task.match(/class/g) || []).length,    // Class count
    task.includes('async') ? 1 : 0,         // Async flag
    task.includes('test') ? 1 : 0,          // Test code flag
    task.split('\n').length / 100           // Normalized line count
  ];
}
```

## Multi-Objective Optimization

```typescript
// Pareto frontier for multiple objectives
interface MultiObjective {
  accuracy: number;
  latency: number;
  cost: number;
}

// Find non-dominated variants
function findParetoFrontier(
  variants: Map<string, MultiObjective>
): string[] {
  const nonDominated: string[] = [];
  
  for (const [id1, obj1] of variants) {
    let isDominated = false;
    
    for (const [id2, obj2] of variants) {
      if (id1 === id2) continue;
      
      // Check if obj2 dominates obj1
      if (
        obj2.accuracy >= obj1.accuracy &&
        obj2.latency <= obj1.latency &&
        obj2.cost <= obj1.cost &&
        (obj2.accuracy > obj1.accuracy ||
         obj2.latency < obj1.latency ||
         obj2.cost < obj1.cost)
      ) {
        isDominated = true;
        break;
      }
    }
    
    if (!isDominated) {
      nonDominated.push(id1);
    }
  }
  
  return nonDominated;
}
```

## Dynamic Exploration Rate

```typescript
// Adaptive exploration based on uncertainty
function adaptiveExploration(
  totalTrials: number,
  recentSuccessRate: number,
  uncertainty: number
): number {
  const baseRate = 0.1;
  const uncertaintyBonus = uncertainty * 0.2;
  const performancePenalty = (1 - recentSuccessRate) * 0.15;
  
  return Math.min(
    0.5,
    baseRate + uncertaintyBonus + performancePenalty
  );
}
```

## Batch Experimentation

```typescript
// Run multiple experiments in parallel
async function batchExperiment(
  tasks: string[],
  variants: string[],
  plugin: ABLearningPlugin
): Promise<void> {
  const results = await Promise.all(
    tasks.map(async (task, i) => {
      const variantId = plugin.thompsonSample(variants);
      const startTime = Date.now();
      
      // Execute task with selected variant
      const result = await executeTask(task, variantId);
      
      const latency = Date.now() - startTime;
      
      // Record feedback
      plugin.recordFeedback({
        taskId: `task_${i}`,
        variantId,
        reward: result.success ? 1.0 : 0.0,
        latencyMs: latency,
        tokenCost: result.tokenCost,
        success: result.success,
        timestamp: Date.now()
      });
      
      return { task, variantId, result };
    })
  );
  
  return results;
}
```

## Curriculum Learning

```typescript
// Gradually increase task difficulty
interface Curriculum {
  level: number;
  minSuccessRate: number;
  taskComplexity: number;
}

const curriculum: Curriculum[] = [
  { level: 1, minSuccessRate: 0.8, taskComplexity: 1 },
  { level: 2, minSuccessRate: 0.7, taskComplexity: 2 },
  { level: 3, minSuccessRate: 0.6, taskComplexity: 3 }
];

function getCurrentLevel(
  successRate: number,
  currentLevel: number
): number {
  const current = curriculum[currentLevel - 1];
  
  if (successRate >= current.minSuccessRate && 
      currentLevel < curriculum.length) {
    return currentLevel + 1;
  }
  
  return currentLevel;
}
```

## Meta-Learning Across Projects

```typescript
// Transfer learning from similar projects
interface ProjectMetadata {
  language: string;
  framework: string;
  domain: string;
}

function transferKnowledge(
  sourcePlugin: ABLearningPlugin,
  targetPlugin: ABLearningPlugin,
  similarity: number
): void {
  // Transfer successful variants with weighted confidence
  const sourceVariants = sourcePlugin.getTopVariants(5);
  
  sourceVariants.forEach(variant => {
    const perf = sourcePlugin.getVariantStats(variant.id);
    
    // Create variant in target with adjusted priors
    const newId = targetPlugin.createVariant(
      variant.template,
      undefined,
      0
    );
    
    // Initialize with transferred knowledge (weighted by similarity)
    const transferAlpha = perf.alpha * similarity;
    const transferBeta = perf.beta * similarity;
    
    // Update target performance with transferred knowledge
    // (Implementation would update database directly)
  });
}
```

## Real-Time Monitoring

```typescript
// Monitor performance in real-time
function startMonitoring(
  plugin: ABLearningPlugin,
  intervalMs: number = 5000
): void {
  setInterval(() => {
    const variants = plugin.getAllVariants();
    
    console.log('\n=== A/B Learning Status ===');
    
    variants.forEach(variant => {
      const perf = plugin.getVariantStats(variant.id);
      const winRate = perf.alpha / (perf.alpha + perf.beta);
      const ci = confidenceInterval(perf.alpha, perf.beta, 0.95);
      
      console.log(`Variant ${variant.id.slice(0, 8)}:`);
      console.log(`  Win Rate: ${(winRate * 100).toFixed(1)}%`);
      console.log(`  95% CI: [${ci.lower.toFixed(3)}, ${ci.upper.toFixed(3)}]`);
      console.log(`  Trials: ${perf.totalTrials}`);
      console.log(`  Avg Latency: ${perf.avgLatencyMs.toFixed(0)}ms`);
      console.log(`  Avg Cost: ${perf.avgTokenCost.toFixed(0)} tokens`);
    });
  }, intervalMs);
}

function confidenceInterval(
  alpha: number,
  beta: number,
  confidence: number
): { lower: number; upper: number } {
  // Use normal approximation for Beta distribution
  const mean = alpha / (alpha + beta);
  const variance = (alpha * beta) / 
    ((alpha + beta) ** 2 * (alpha + beta + 1));
  const stdDev = Math.sqrt(variance);
  const z = 1.96; // 95% confidence
  
  return {
    lower: Math.max(0, mean - z * stdDev),
    upper: Math.min(1, mean + z * stdDev)
  };
}
```

## A/B/n Testing (Multiple Variants)

```typescript
// Test more than two variants simultaneously
async function runABnTest(
  variants: string[],
  nTrials: number,
  plugin: ABLearningPlugin
): Promise<string> {
  for (let i = 0; i < nTrials; i++) {
    const selected = plugin.thompsonSample(variants);
    
    // Execute and record feedback
    const result = await executeWithVariant(selected);
    
    plugin.recordFeedback({
      taskId: `trial_${i}`,
      variantId: selected,
      reward: result.reward,
      latencyMs: result.latencyMs,
      tokenCost: result.tokenCost,
      success: result.success,
      timestamp: Date.now()
    });
  }
  
  // Find overall winner
  const evaluations = [];
  for (let i = 0; i < variants.length; i++) {
    for (let j = i + 1; j < variants.length; j++) {
      const result = plugin.evaluateABTest(
        variants[i],
        variants[j]
      );
      evaluations.push(result);
    }
  }
  
  // Return variant with most wins
  const winCounts = new Map<string, number>();
  evaluations.forEach(eval => {
    if (eval.winner) {
      winCounts.set(
        eval.winner,
        (winCounts.get(eval.winner) || 0) + 1
      );
    }
  });
  
  return Array.from(winCounts.entries())
    .sort((a, b) => b[1] - a[1])[0][0];
}
```
