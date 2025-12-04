# OpenCode A/B Learning Plugin

A sophisticated self-learning plugin for [OpenCode](https://opencode.ai) that implements A/B testing with Thompson Sampling, genetic algorithm evolution, and RLHF-inspired feedback loops.

## Features

🧪 **A/B Testing Pipeline**
- Multi-armed bandit optimization with Thompson Sampling
- Bayesian posterior distribution updates
- Statistical significance testing with confidence intervals

🧬 **Genetic Algorithm Evolution**
- Automatic prompt variant evolution through LLM-based crossover
- Adaptive mutation rates
- Multi-generation optimization

🎯 **Contextual Bandits**
- LinUCB selection with rich context features
- LLM-enhanced feature extraction
- Uncertainty-aware exploration

📊 **RLHF-Style Learning**
- Human preference feedback integration
- Pairwise comparison tracking
- Bradley-Terry model updates

🔄 **Continuous Learning**
- Implicit reward signals from execution
- Automatic performance tracking
- Background optimization during idle time

## Installation

### Local Project Installation

```bash
# In your OpenCode project
mkdir -p .opencode/plugin
cd .opencode/plugin
git clone https://github.com/d-oit/opencode-ab-learning-plugin.git
ln -s opencode-ab-learning-plugin/src/ab-learning.ts ab-learning.ts
```

### Global Installation

```bash
# For all OpenCode projects
mkdir -p ~/.config/opencode/plugin
cd ~/.config/opencode/plugin
git clone https://github.com/d-oit/opencode-ab-learning-plugin.git
ln -s opencode-ab-learning-plugin/src/ab-learning.ts ab-learning.ts
```

## Usage

Once installed, the plugin automatically activates when you start OpenCode:

```bash
cd your-project
opencode init # If first time
opencode
```

### Automatic A/B Testing

The plugin automatically runs experiments on your tasks:

```
You: "Optimize this authentication function"
```

The plugin will:
1. Generate prompt variants
2. Use Thompson Sampling to select the best approach
3. Track performance metrics (success rate, latency, token cost)
4. Learn from outcomes

### Check Experiment Results

```
You: "Show A/B test results for authentication tasks"
```

### Manual Variant Testing

```
You: "Test these three database approaches: PostgreSQL, MongoDB, Redis"
```

### Evolve Prompts

```
You: "Evolve better code review prompts"
```

## Architecture

### Database Schema

The plugin uses SQLite with the following tables:

- `prompt_variants` - Stores prompt templates and genealogy
- `variant_performance` - Tracks Bayesian statistics (alpha/beta)
- `experiment_assignments` - Maps tasks to variants
- `preference_comparisons` - RLHF pairwise preferences
- `contextual_features` - Context-aware performance data

### Learning Loop

```
┌─────────────────┐
│ User Request    │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Thompson Sampling   │ ◄── Bayesian posteriors
│ Variant Selection   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Execute Task        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Record Feedback     │ ◄── Implicit + Explicit
│ Update Statistics   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Evaluate & Evolve   │ ◄── Genetic algorithm
│ (Background)        │
└─────────────────────┘
```

## Configuration

The plugin automatically adjusts:

- **Exploration Rate**: Starts at 0.3, decays to 0.05
- **Temperature**: Starts at 1.0 for action selection
- **Min Trials**: 10 trials before exploitation begins

## Tools Available

The plugin exposes these tools to OpenCode:

1. **Thompson Sample** - Select optimal variant with exploration
2. **Record Feedback** - Update performance with rewards
3. **Evaluate A/B Test** - Check statistical significance
4. **Evolve Prompts** - Generate new variants via GA
5. **Contextual Bandit** - Context-aware variant selection

## Performance

- **Overhead**: ~10-50ms per decision
- **Database**: ~1MB per 1000 experiments
- **Token Optimization**: Built-in reward weighting
- **Auto-pruning**: Low-performing variants removed during idle

## Research Background

This plugin implements cutting-edge techniques from:

- **Thompson Sampling** (Agrawal & Goyal, 2012) - Bayesian bandits
- **LinUCB** (Li et al., 2010) - Contextual bandits with linear payoff
- **SEAL** (Chen et al., 2024) - Self-improving LLM agents
- **Genetic Algorithms** - Population-based optimization
- **RLHF** (Christiano et al., 2017) - Human feedback integration

## Development

```bash
# Run tests
bun test

# Check types
bun run typecheck

# Format code
bun run format
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file

## Author

Dominik Oswald ([@d-oit](https://github.com/d-oit))

## Links

- [OpenCode Documentation](https://opencode.ai/docs)
- [Plugin API Reference](https://opencode.ai/docs/plugins)
- [Issue Tracker](https://github.com/d-oit/opencode-ab-learning-plugin/issues)
