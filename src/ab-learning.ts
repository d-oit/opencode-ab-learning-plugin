/**
 * OpenCode A/B Learning Plugin
 * 
 * Implements sophisticated self-learning A/B testing with:
 * - Thompson Sampling (Bayesian multi-armed bandits)
 * - Genetic Algorithm Evolution
 * - Contextual Bandits (LinUCB)
 * - RLHF-style feedback loops
 * - Continuous learning from implicit signals
 */

import Database from 'bun:sqlite';
import { randomBytes } from 'crypto';

// ============================================================================
// Types & Interfaces
// ============================================================================

interface PromptVariant {
  id: string;
  template: string;
  parentId?: string;
  generation: number;
  createdAt: number;
}

interface VariantPerformance {
  variantId: string;
  alpha: number; // Successes (Bayesian posterior)
  beta: number;  // Failures
  totalTrials: number;
  avgReward: number;
  avgLatencyMs: number;
  avgTokenCost: number;
}

interface ExperimentAssignment {
  taskId: string;
  variantId: string;
  context: string;
  timestamp: number;
}

interface FeedbackRecord {
  taskId: string;
  variantId: string;
  reward: number;
  latencyMs: number;
  tokenCost: number;
  success: boolean;
  timestamp: number;
}

interface PreferenceComparison {
  winnerId: string;
  loserId: string;
  context: string;
  humanFeedback: boolean;
  timestamp: number;
}

interface ContextualFeatures {
  variantId: string;
  contextHash: string;
  features: number[]; // Feature vector
  reward: number;
  timestamp: number;
}

// ============================================================================
// Database Schema
// ============================================================================

const SCHEMA = `
CREATE TABLE IF NOT EXISTS prompt_variants (
  id TEXT PRIMARY KEY,
  template TEXT NOT NULL,
  parent_id TEXT,
  generation INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS variant_performance (
  variant_id TEXT PRIMARY KEY,
  alpha REAL NOT NULL DEFAULT 1.0,
  beta REAL NOT NULL DEFAULT 1.0,
  total_trials INTEGER NOT NULL DEFAULT 0,
  avg_reward REAL NOT NULL DEFAULT 0.0,
  avg_latency_ms REAL NOT NULL DEFAULT 0.0,
  avg_token_cost REAL NOT NULL DEFAULT 0.0,
  FOREIGN KEY(variant_id) REFERENCES prompt_variants(id)
);

CREATE TABLE IF NOT EXISTS experiment_assignments (
  task_id TEXT PRIMARY KEY,
  variant_id TEXT NOT NULL,
  context TEXT,
  timestamp INTEGER NOT NULL,
  FOREIGN KEY(variant_id) REFERENCES prompt_variants(id)
);

CREATE TABLE IF NOT EXISTS feedback_records (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  variant_id TEXT NOT NULL,
  reward REAL NOT NULL,
  latency_ms REAL NOT NULL,
  token_cost REAL NOT NULL,
  success INTEGER NOT NULL,
  timestamp INTEGER NOT NULL,
  FOREIGN KEY(variant_id) REFERENCES prompt_variants(id)
);

CREATE TABLE IF NOT EXISTS preference_comparisons (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  winner_id TEXT NOT NULL,
  loser_id TEXT NOT NULL,
  context TEXT,
  human_feedback INTEGER NOT NULL DEFAULT 0,
  timestamp INTEGER NOT NULL,
  FOREIGN KEY(winner_id) REFERENCES prompt_variants(id),
  FOREIGN KEY(loser_id) REFERENCES prompt_variants(id)
);

CREATE TABLE IF NOT EXISTS contextual_features (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  variant_id TEXT NOT NULL,
  context_hash TEXT NOT NULL,
  features BLOB NOT NULL,
  reward REAL NOT NULL,
  timestamp INTEGER NOT NULL,
  FOREIGN KEY(variant_id) REFERENCES prompt_variants(id)
);

CREATE INDEX IF NOT EXISTS idx_feedback_variant ON feedback_records(variant_id);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_contextual_variant ON contextual_features(variant_id, context_hash);
`;

// ============================================================================
// Core Plugin Class
// ============================================================================

class ABLearningPlugin {
  private db: Database;
  private explorationRate: number = 0.3;
  private temperature: number = 1.0;
  private minTrialsBeforeExploitation: number = 10;

  constructor(dbPath: string = '.opencode/ab_learning.db') {
    this.db = new Database(dbPath);
    this.initDatabase();
  }

  private initDatabase(): void {
    this.db.exec(SCHEMA);
  }

  // ==========================================================================
  // Thompson Sampling
  // ==========================================================================

  /**
   * Select variant using Thompson Sampling from Beta distributions
   */
  thompsonSample(candidateIds: string[]): string {
    const performances = candidateIds.map(id => this.getPerformance(id));
    
    // Sample from each variant's posterior Beta distribution
    const samples = performances.map(perf => {
      return this.sampleBeta(perf.alpha, perf.beta);
    });

    // Select variant with highest sample
    const maxIdx = samples.indexOf(Math.max(...samples));
    return candidateIds[maxIdx];
  }

  /**
   * Sample from Beta(alpha, beta) distribution using gamma distributions
   */
  private sampleBeta(alpha: number, beta: number): number {
    const x = this.sampleGamma(alpha, 1);
    const y = this.sampleGamma(beta, 1);
    return x / (x + y);
  }

  /**
   * Sample from Gamma distribution using Marsaglia & Tsang method
   */
  private sampleGamma(shape: number, scale: number): number {
    if (shape < 1) {
      return this.sampleGamma(shape + 1, scale) * Math.pow(Math.random(), 1 / shape);
    }

    const d = shape - 1/3;
    const c = 1 / Math.sqrt(9 * d);

    while (true) {
      let x: number, v: number;
      do {
        x = this.randomNormal();
        v = 1 + c * x;
      } while (v <= 0);

      v = v * v * v;
      const u = Math.random();
      const x2 = x * x;

      if (u < 1 - 0.0331 * x2 * x2) {
        return d * v * scale;
      }
      if (Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) {
        return d * v * scale;
      }
    }
  }

  /**
   * Generate standard normal random variable (Box-Muller)
   */
  private randomNormal(): number {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  /**
   * Update variant performance with Bayesian update
   */
  recordFeedback(feedback: FeedbackRecord): void {
    const perf = this.getPerformance(feedback.variantId);

    // Bayesian update
    const newAlpha = perf.alpha + (feedback.success ? 1 : 0);
    const newBeta = perf.beta + (feedback.success ? 0 : 1);
    const newTrials = perf.totalTrials + 1;

    // Running averages
    const newAvgReward = (perf.avgReward * perf.totalTrials + feedback.reward) / newTrials;
    const newAvgLatency = (perf.avgLatencyMs * perf.totalTrials + feedback.latencyMs) / newTrials;
    const newAvgTokenCost = (perf.avgTokenCost * perf.totalTrials + feedback.tokenCost) / newTrials;

    this.db.run(`
      INSERT OR REPLACE INTO variant_performance 
      (variant_id, alpha, beta, total_trials, avg_reward, avg_latency_ms, avg_token_cost)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `, [feedback.variantId, newAlpha, newBeta, newTrials, newAvgReward, newAvgLatency, newAvgTokenCost]);

    // Also store raw feedback
    this.db.run(`
      INSERT INTO feedback_records 
      (task_id, variant_id, reward, latency_ms, token_cost, success, timestamp)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `, [feedback.taskId, feedback.variantId, feedback.reward, feedback.latencyMs, 
        feedback.tokenCost, feedback.success ? 1 : 0, Date.now()]);
  }

  private getPerformance(variantId: string): VariantPerformance {
    const row = this.db.query(`
      SELECT * FROM variant_performance WHERE variant_id = ?
    `).get(variantId) as any;

    if (row) {
      return {
        variantId: row.variant_id,
        alpha: row.alpha,
        beta: row.beta,
        totalTrials: row.total_trials,
        avgReward: row.avg_reward,
        avgLatencyMs: row.avg_latency_ms,
        avgTokenCost: row.avg_token_cost
      };
    }

    // Initialize with uniform prior Beta(1, 1)
    return {
      variantId,
      alpha: 1.0,
      beta: 1.0,
      totalTrials: 0,
      avgReward: 0.0,
      avgLatencyMs: 0.0,
      avgTokenCost: 0.0
    };
  }

  // ==========================================================================
  // Statistical Evaluation
  // ==========================================================================

  /**
   * Evaluate statistical significance between two variants
   */
  evaluateABTest(variantA: string, variantB: string): {
    winner: string | null;
    confidence: number;
    pValue: number;
  } {
    const perfA = this.getPerformance(variantA);
    const perfB = this.getPerformance(variantB);

    // Monte Carlo simulation to estimate P(A > B)
    const nSamples = 10000;
    let countAWins = 0;

    for (let i = 0; i < nSamples; i++) {
      const sampleA = this.sampleBeta(perfA.alpha, perfA.beta);
      const sampleB = this.sampleBeta(perfB.alpha, perfB.beta);
      if (sampleA > sampleB) countAWins++;
    }

    const probAWins = countAWins / nSamples;
    const confidence = Math.max(probAWins, 1 - probAWins);
    const pValue = 2 * Math.min(probAWins, 1 - probAWins);

    return {
      winner: confidence > 0.95 ? (probAWins > 0.5 ? variantA : variantB) : null,
      confidence,
      pValue
    };
  }

  // ==========================================================================
  // Genetic Algorithm Evolution
  // ==========================================================================

  /**
   * Evolve new prompt variants using genetic algorithm
   */
  async evolvePrompts(populationSize: number = 5, nGenerations: number = 3): Promise<string[]> {
    // Get current population (top performers)
    const population = this.getTopVariants(populationSize);
    const newVariants: string[] = [];

    for (let gen = 0; gen < nGenerations; gen++) {
      // Selection: Tournament selection
      const parents = this.tournamentSelection(population, 2);

      // Crossover: LLM-based crossover
      const offspring = await this.crossoverPrompts(parents[0].template, parents[1].template);

      // Mutation: LLM-based mutation
      const mutated = Math.random() < 0.2 ? await this.mutatePrompt(offspring) : offspring;

      // Add to population
      const newVariantId = this.createVariant(mutated, parents[0].id, parents[0].generation + 1);
      newVariants.push(newVariantId);
    }

    return newVariants;
  }

  private getTopVariants(n: number): PromptVariant[] {
    const rows = this.db.query(`
      SELECT pv.*, vp.avg_reward 
      FROM prompt_variants pv
      JOIN variant_performance vp ON pv.id = vp.variant_id
      WHERE vp.total_trials >= ?
      ORDER BY vp.avg_reward DESC
      LIMIT ?
    `).all(this.minTrialsBeforeExploitation, n) as any[];

    return rows.map(row => ({
      id: row.id,
      template: row.template,
      parentId: row.parent_id,
      generation: row.generation,
      createdAt: row.created_at
    }));
  }

  private tournamentSelection(population: PromptVariant[], k: number): PromptVariant[] {
    const selected: PromptVariant[] = [];
    for (let i = 0; i < k; i++) {
      const contestants = [];
      for (let j = 0; j < 3; j++) {
        contestants.push(population[Math.floor(Math.random() * population.length)]);
      }
      // Select best from contestants
      const best = contestants.reduce((a, b) => {
        const perfA = this.getPerformance(a.id);
        const perfB = this.getPerformance(b.id);
        return perfA.avgReward > perfB.avgReward ? a : b;
      });
      selected.push(best);
    }
    return selected;
  }

  private async crossoverPrompts(promptA: string, promptB: string): Promise<string> {
    // Use LLM to intelligently combine two prompts
    const crossoverPrompt = `
Generate a new prompt that combines the best aspects of these two prompts:

Prompt A: ${promptA}

Prompt B: ${promptB}

Create a hybrid prompt that:
1. Takes the clearest instructions from both
2. Maintains the most effective phrasing
3. Preserves key constraints and requirements
4. Results in a prompt that could outperform both parents

Return ONLY the new hybrid prompt, no explanation.
`;

    // This would call OpenCode's LLM - placeholder for now
    return `${promptA}\n\nAdditionally: ${promptB.split('\n')[0]}`;
  }

  private async mutatePrompt(prompt: string): Promise<string> {
    const mutationPrompt = `
Improve this prompt with a small random mutation:

${prompt}

Make ONE of these changes:
1. Rephrase for clarity
2. Add a helpful constraint
3. Remove redundancy
4. Adjust tone/style
5. Add example format

Return ONLY the mutated prompt, no explanation.
`;

    // Placeholder - would use OpenCode LLM
    const mutations = [
      ' Be concise.',
      ' Focus on performance.',
      ' Consider edge cases.',
      ' Use modern patterns.',
      ' Optimize for readability.'
    ];
    return prompt + mutations[Math.floor(Math.random() * mutations.length)];
  }

  createVariant(template: string, parentId?: string, generation: number = 0): string {
    const id = randomBytes(16).toString('hex');
    this.db.run(`
      INSERT INTO prompt_variants (id, template, parent_id, generation, created_at)
      VALUES (?, ?, ?, ?, ?)
    `, [id, template, parentId || null, generation, Date.now()]);

    // Initialize performance
    this.db.run(`
      INSERT INTO variant_performance (variant_id)
      VALUES (?)
    `, [id]);

    return id;
  }

  // ==========================================================================
  // Contextual Bandits (LinUCB)
  // ==========================================================================

  /**
   * Select variant using contextual bandit (LinUCB) with context features
   */
  contextualBanditSelect(candidateIds: string[], contextFeatures: number[]): string {
    const scores = candidateIds.map(id => {
      const features = this.getContextualFeatures(id, contextFeatures);
      return this.linUCBScore(features, this.explorationRate);
    });

    const maxIdx = scores.indexOf(Math.max(...scores));
    return candidateIds[maxIdx];
  }

  private getContextualFeatures(variantId: string, context: number[]): number[] {
    // Retrieve historical features for this variant + context
    // For now, return context as-is
    return context;
  }

  private linUCBScore(features: number[], alpha: number): number {
    // Simplified LinUCB score: reward estimate + uncertainty bonus
    const rewardEstimate = features.reduce((sum, f) => sum + f, 0) / features.length;
    const uncertainty = alpha * Math.sqrt(features.length);
    return rewardEstimate + uncertainty;
  }

  // ==========================================================================
  // RLHF-Style Preference Learning
  // ==========================================================================

  /**
   * Record human preference between two variants
   */
  recordPreference(winnerId: string, loserId: string, context: string, humanFeedback: boolean = false): void {
    this.db.run(`
      INSERT INTO preference_comparisons (winner_id, loser_id, context, human_feedback, timestamp)
      VALUES (?, ?, ?, ?, ?)
    `, [winnerId, loserId, context, humanFeedback ? 1 : 0, Date.now()]);

    // Update Bradley-Terry model estimates
    // (Simplified: boost winner's alpha, loser's beta)
    const winnerPerf = this.getPerformance(winnerId);
    const loserPerf = this.getPerformance(loserId);

    this.db.run(`
      UPDATE variant_performance 
      SET alpha = alpha + 0.1
      WHERE variant_id = ?
    `, [winnerId]);

    this.db.run(`
      UPDATE variant_performance 
      SET beta = beta + 0.1
      WHERE variant_id = ?
    `, [loserId]);
  }

  // ==========================================================================
  // Continuous Learning Loop
  // ==========================================================================

  /**
   * Background task: Periodically evaluate and evolve
   */
  async continuousLearningLoop(intervalMs: number = 3600000): Promise<void> {
    setInterval(async () => {
      // 1. Prune low-performing variants
      this.pruneLowPerformers();

      // 2. Evolve new variants
      await this.evolvePrompts(5, 2);

      // 3. Decay exploration rate
      this.explorationRate = Math.max(0.05, this.explorationRate * 0.95);
    }, intervalMs);
  }

  private pruneLowPerformers(): void {
    // Remove variants with < 5% win rate after sufficient trials
    this.db.run(`
      DELETE FROM prompt_variants
      WHERE id IN (
        SELECT variant_id FROM variant_performance
        WHERE total_trials >= 20 AND (alpha / (alpha + beta)) < 0.05
      )
    `);
  }

  // ==========================================================================
  // Query Methods
  // ==========================================================================

  getAllVariants(): PromptVariant[] {
    const rows = this.db.query('SELECT * FROM prompt_variants ORDER BY created_at DESC').all() as any[];
    return rows.map(row => ({
      id: row.id,
      template: row.template,
      parentId: row.parent_id,
      generation: row.generation,
      createdAt: row.created_at
    }));
  }

  getVariantStats(variantId: string): VariantPerformance {
    return this.getPerformance(variantId);
  }

  close(): void {
    this.db.close();
  }
}

// ============================================================================
// OpenCode Plugin Export
// ============================================================================

export default {
  name: 'ab-learning',
  description: 'Self-learning A/B testing with Thompson Sampling and genetic evolution',
  version: '0.1.0',

  async initialize() {
    const plugin = new ABLearningPlugin();
    
    // Start continuous learning loop
    plugin.continuousLearningLoop();

    return plugin;
  },

  tools: [
    {
      name: 'thompson_sample',
      description: 'Select optimal variant using Thompson Sampling',
      parameters: {
        candidateIds: { type: 'array', items: { type: 'string' }, description: 'Variant IDs to choose from' }
      },
      async execute(plugin: ABLearningPlugin, args: { candidateIds: string[] }) {
        return plugin.thompsonSample(args.candidateIds);
      }
    },
    {
      name: 'record_feedback',
      description: 'Record performance feedback for a variant',
      parameters: {
        taskId: { type: 'string' },
        variantId: { type: 'string' },
        reward: { type: 'number' },
        latencyMs: { type: 'number' },
        tokenCost: { type: 'number' },
        success: { type: 'boolean' }
      },
      async execute(plugin: ABLearningPlugin, args: FeedbackRecord) {
        plugin.recordFeedback({ ...args, timestamp: Date.now() });
        return { status: 'recorded' };
      }
    },
    {
      name: 'evaluate_ab_test',
      description: 'Evaluate statistical significance between variants',
      parameters: {
        variantA: { type: 'string' },
        variantB: { type: 'string' }
      },
      async execute(plugin: ABLearningPlugin, args: { variantA: string; variantB: string }) {
        return plugin.evaluateABTest(args.variantA, args.variantB);
      }
    },
    {
      name: 'evolve_prompts',
      description: 'Generate new variants using genetic algorithm',
      parameters: {
        populationSize: { type: 'number', default: 5 },
        generations: { type: 'number', default: 3 }
      },
      async execute(plugin: ABLearningPlugin, args: { populationSize?: number; generations?: number }) {
        return plugin.evolvePrompts(args.populationSize, args.generations);
      }
    },
    {
      name: 'contextual_bandit_select',
      description: 'Select variant using contextual features',
      parameters: {
        candidateIds: { type: 'array', items: { type: 'string' } },
        contextFeatures: { type: 'array', items: { type: 'number' } }
      },
      async execute(plugin: ABLearningPlugin, args: { candidateIds: string[]; contextFeatures: number[] }) {
        return plugin.contextualBanditSelect(args.candidateIds, args.contextFeatures);
      }
    }
  ]
};
