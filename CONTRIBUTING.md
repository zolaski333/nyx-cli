# Contributing to Nyx

Thank you for your interest in contributing to Nyx! We welcome contributions of all kinds — bug fixes, features, documentation, and ideas.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/nyx-cli/nyx.git
cd nyx

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Verify everything works
nyx --help
```

## 🧪 Development

### Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check code style
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Type Checking

We use [mypy](https://mypy-lang.org/) for static type checking:

```bash
mypy nyx/
```

### Testing

We use [pytest](https://docs.pytest.org/):

```bash
pytest
```

## 📋 Pull Request Guidelines

1. **Fork the repository** and create your branch from `main`.
2. **Run the tests** before submitting: `pytest`
3. **Run the linter**: `ruff check .`
4. **Run type checking**: `mypy nyx/`
5. **Keep changes focused** — one feature/fix per PR.
6. **Write clear commit messages** following [conventional commits](https://www.conventionalcommits.org/).
7. **Update documentation** if you change functionality.

## 🐛 Reporting Issues

- Use the [GitHub issue tracker](https://github.com/nyx-cli/nyx/issues)
- Include your Python version (`python --version`)
- Include the output of `nyx --help`
- Include steps to reproduce the issue
- Include the full error output

## 🎯 Feature Requests

We're particularly interested in:

- New LLM providers
- MCP server integrations
- Skill examples and templates
- Performance improvements
- Documentation improvements
- CI/CD integrations

## 📝 Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to abide by its terms.

## 🙏 Thank You!

Every contribution, no matter how small, makes Nyx better for everyone. Thank you! ❤️