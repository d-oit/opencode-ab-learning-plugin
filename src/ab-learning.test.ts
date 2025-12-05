import { beforeEach, describe, expect, test } from 'bun:test';
import { unlink } from 'fs/promises';
import Database from 'bun:sqlite';
import pluginModule from './ab-learning';

const TEST_DB = '.opencode/test_ab_learning.db';

// Narrow the default export to the expected shape
const pluginFactory = pluginModule as unknown as {
  initialize: () => Promise<any>;
};

describe('ABLearningPlugin integration', () => {
  beforeEach(async () => {
    try {
      await unlink(TEST_DB);
    } catch {
      // ignore
    }
  });

  test('initializes and creates database schema', async () => {
    const plugin = await pluginFactory.initialize();

    const db = new Database(TEST_DB);
    const tables = db
      .query<[{ name: string }]>(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
      )
      .all();

    const tableNames = tables.map(t => t.name);

    expect(tableNames).toContain('prompt_variants');
    expect(tableNames).toContain('variant_performance');
    expect(tableNames).toContain('feedback_records');
    expect(tableNames).toContain('preference_comparisons');

    db.close();
    plugin.close?.();
  });

  test('Thompson Sampling prefers higher alpha/beta ratio', async () => {
    const plugin = await pluginFactory.initialize();

    const lowId = plugin.createVariant('low performer');
    const highId = plugin.createVariant('high performer');

    // Simulate feedback: high performer wins more often
    for (let i = 0; i < 20; i++) {
      plugin.recordFeedback({
        taskId: `t_high_${i}`,
        variantId: highId,
        reward: 1,
        latencyMs: 100,
        tokenCost: 50,
        success: true,
        timestamp: Date.now(),
      });
    }

    for (let i = 0; i < 5; i++) {
      plugin.recordFeedback({
        taskId: `t_low_${i}`,
        variantId: lowId,
        reward: 0,
        latencyMs: 200,
        tokenCost: 80,
        success: false,
        timestamp: Date.now(),
      });
    }

    let highWins = 0;
    const trials = 50;

    for (let i = 0; i < trials; i++) {
      const chosen = plugin.thompsonSample([lowId, highId]);
      if (chosen === highId) highWins++;
    }

    // High performer should be chosen most of the time
    expect(highWins).toBeGreaterThan(trials * 0.7);

    plugin.close();
  });

  test('records feedback and updates Bayesian statistics', async () => {
    const plugin = await pluginFactory.initialize();
    const variantId = plugin.createVariant('test variant');

    plugin.recordFeedback({
      taskId: 'task_1',
      variantId,
      reward: 1,
      latencyMs: 150,
      tokenCost: 100,
      success: true,
      timestamp: Date.now(),
    });

    const stats = plugin.getVariantStats(variantId);

    expect(stats.totalTrials).toBe(1);
    expect(stats.alpha).toBeGreaterThan(1); // prior alpha was 1
    expect(stats.beta).toBe(1); // no failures yet
    expect(stats.avgReward).toBeGreaterThan(0.5);

    plugin.close();
  });

  test('evaluateABTest returns sensible winner and confidence', async () => {
    const plugin = await pluginFactory.initialize();

    const a = plugin.createVariant('A');
    const b = plugin.createVariant('B');

    // Variant A: mostly wins
    for (let i = 0; i < 30; i++) {
      plugin.recordFeedback({
        taskId: `A_${i}`,
        variantId: a,
        reward: 1,
        latencyMs: 120,
        tokenCost: 90,
        success: true,
        timestamp: Date.now(),
      });
    }

    // Variant B: mostly loses
    for (let i = 0; i < 30; i++) {
      plugin.recordFeedback({
        taskId: `B_${i}`,
        variantId: b,
        reward: 0,
        latencyMs: 200,
        tokenCost: 120,
        success: false,
        timestamp: Date.now(),
      });
    }

    const result = plugin.evaluateABTest(a, b);

    expect(result.winner).toBe(a);
    expect(result.confidence).toBeGreaterThan(0.9);

    plugin.close();
  });
});
