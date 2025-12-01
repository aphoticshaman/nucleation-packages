# Contributing to Nucleation

Thank you for your interest in contributing! This document provides guidelines and steps for contributing.

## Code of Conduct

Be respectful. Be constructive. We're all here to build something useful.

## Getting Started

### Prerequisites

- Node.js 18+
- npm 9+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/aphoticshaman/nucleation-packages.git
cd nucleation-packages

# Install dependencies
npm install

# Build all packages
npm run build

# Run tests
npm test
```

### Project Structure

```
nucleation-packages/
├── packages/
│   ├── core/           # Shared WASM initialization and base classes
│   ├── nucleation/     # Meta-package with unified API
│   ├── regime-shift/   # Finance detector
│   ├── threat-pulse/   # Security detector
│   └── ...             # Other domain detectors
├── harness/            # Integration tests and examples
├── docs/               # Documentation
└── .github/            # CI/CD workflows
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

- Write TypeScript code
- Follow existing patterns
- Add tests for new functionality
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run all tests
npm test

# Run tests for a specific package
npm test -w packages/regime-shift

# Run with coverage
npm run test:coverage

# Type check
npm run typecheck

# Lint
npm run lint
```

### 4. Commit

We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: type(scope): description
git commit -m "feat(regime-shift): add multi-asset correlation"
git commit -m "fix(core): handle NaN values gracefully"
git commit -m "docs: update API reference"
git commit -m "test(threat-pulse): add batch processing tests"
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Build process or auxiliary tool changes

### 5. Push and Create PR

```bash
git push origin your-branch-name
```

Then create a Pull Request on GitHub.

## Pull Request Guidelines

### Requirements

- [ ] Tests pass (`npm test`)
- [ ] Linting passes (`npm run lint`)
- [ ] Type checking passes (`npm run typecheck`)
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG entry added (for user-facing changes)

### PR Description Template

```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## Adding a New Detector

To add a new domain-specific detector:

1. **Create package structure:**
   ```bash
   mkdir -p packages/your-detector/{src,test,examples}
   ```

2. **Create package.json:**
   ```json
   {
     "name": "your-detector",
     "version": "1.0.0",
     "type": "module",
     "main": "dist/index.js",
     "types": "dist/index.d.ts",
     "dependencies": {
       "@nucleation/core": "workspace:*"
     }
   }
   ```

3. **Implement using BaseDetector:**
   ```typescript
   import { BaseDetector, type DetectorConfig } from '@nucleation/core';

   export class YourDetector extends BaseDetector {
     protected mapPhaseToLevel(phase: Phase): string {
       // Map phases to domain-specific levels
     }
   }
   ```

4. **Add tests**

5. **Update root tsconfig.json references**

6. **Update nucleation meta-package exports**

## Coding Standards

### TypeScript

- Use strict mode
- Prefer explicit types over inference for public APIs
- Use `readonly` where appropriate
- Avoid `any` - use `unknown` if needed

### Naming

- Classes: PascalCase
- Functions/variables: camelCase
- Constants: SCREAMING_SNAKE_CASE
- Files: kebab-case

### Comments

- Document public APIs with JSDoc
- Explain "why", not "what"
- Keep comments up-to-date

## Questions?

- Open a [Discussion](https://github.com/aphoticshaman/nucleation-packages/discussions)
- Check existing [Issues](https://github.com/aphoticshaman/nucleation-packages/issues)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
