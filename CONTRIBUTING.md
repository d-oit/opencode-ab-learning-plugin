# Contributing to OpenCode A/B Learning Plugin

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/opencode-ab-learning-plugin.git
   cd opencode-ab-learning-plugin
   ```

2. **Install Bun** (if not already installed)
   ```bash
   curl -fsSL https://bun.sh/install | bash
   ```

3. **Install Dependencies**
   ```bash
   bun install
   ```

4. **Run Tests**
   ```bash
   bun test
   ```

## Project Structure

```
opencode-ab-learning-plugin/
├── src/
│   ├── ab-learning.ts       # Main plugin implementation
│   └── ab-learning.test.ts  # Test suite
├── examples/
│   └── basic-usage.md       # Usage examples
├── .github/
│   └── workflows/
│       └── ci.yml           # CI/CD pipeline
├── package.json
├── tsconfig.json
├── README.md
└── CONTRIBUTING.md
```

## Making Changes

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clear, documented code
   - Follow TypeScript best practices
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   bun test
   bun run typecheck
   bun run format
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

   Follow conventional commit messages:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Test additions/changes
   - `refactor:` - Code refactoring
   - `perf:` - Performance improvements
   - `chore:` - Maintenance tasks

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## Code Style

- Use TypeScript strict mode
- Follow functional programming principles where possible
- Document all public APIs with JSDoc comments
- Keep functions small and focused
- Use meaningful variable names

## Testing Guidelines

- Write tests for all new features
- Ensure tests are isolated and deterministic
- Use descriptive test names
- Test edge cases and error conditions
- Aim for high code coverage

## Areas for Contribution

### High Priority
- [ ] Implement comprehensive test suite
- [ ] Add integration tests with OpenCode
- [ ] Performance benchmarks
- [ ] LLM-based crossover/mutation implementation
- [ ] Advanced contextual feature extraction

### Medium Priority
- [ ] Web dashboard for experiment visualization
- [ ] Export/import experiment data
- [ ] Multi-objective optimization (Pareto frontier)
- [ ] A/B/n testing support (> 2 variants)
- [ ] Bayesian optimization for hyperparameters

### Nice to Have
- [ ] Plugin configuration UI
- [ ] Real-time monitoring dashboard
- [ ] Integration with external analytics
- [ ] Docker container for isolated testing
- [ ] CLI tool for experiment management

## Research Improvements

If you're interested in advancing the research aspects:

1. **Thompson Sampling Variants**
   - Implement contextual Thompson Sampling
   - Add support for non-binary rewards
   - Explore discounted rewards for non-stationary environments

2. **Genetic Algorithm Enhancements**
   - Implement novelty search
   - Add multi-objective optimization (NSGA-II)
   - Explore Quality-Diversity algorithms (MAP-Elites)

3. **Contextual Bandits**
   - Implement neural contextual bandits
   - Add deep LinUCB
   - Explore kernel-based methods

4. **RLHF Improvements**
   - Implement direct preference optimization (DPO)
   - Add reward modeling
   - Explore constitutional AI principles

## Documentation

When adding new features:

1. Update README.md with usage examples
2. Add JSDoc comments to all public methods
3. Create examples in `examples/` directory
4. Update architecture diagrams if needed

## Questions?

Feel free to:
- Open an issue for discussion
- Ask questions in PR comments
- Reach out to [@d-oit](https://github.com/d-oit)

## Code of Conduct

Be respectful, inclusive, and professional. We're all here to learn and improve the project together.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
