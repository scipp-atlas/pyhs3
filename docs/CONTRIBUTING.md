# Contributing to pyhs3

We welcome contributions to pyhs3! This guide will help you get started with
contributing to the project.

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.10 or higher
- Git
- A GitHub account

### Setting Up Your Development Environment

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR-USERNAME/pyhs3.git
   cd pyhs3
   ```

2. **Create a development environment**

   We recommend using a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the package in development mode**

   ```bash
   pip install -e .[dev,test,docs]
   ```

4. **Set up pre-commit hooks**

   We use pre-commit to ensure code quality:

   ```bash
   pip install pre-commit
   pre-commit install
   ```

   This will automatically run linting and formatting checks before each commit.

## Code Standards

### Code Style

We follow strict code quality standards:

- **Formatting**: We use `ruff format` for code formatting
- **Linting**: We use `ruff check` for linting
- **Type checking**: We use `mypy` for static type checking
- **Docstring style**: We follow numpy-style docstrings checked with
  `pydocstyle`

All of these are enforced through pre-commit hooks.

### Type Hints

All code must include type hints. We enforce this through mypy with strict
settings:

```python
from __future__ import annotations


def calculate_logpdf(x: float, mean: float, sigma: float) -> float:
    """Calculate log PDF of a Gaussian distribution."""
    ...
```

### Testing

All contributions must include tests. See [testing](testing.rst) for detailed
information on writing and running tests.

### Documentation

- All public APIs must have docstrings
- Update relevant documentation files when adding features
- Use numpy-style docstrings

## Git Workflow

### Branch Naming

Create descriptive branch names:

- `feat/feature-name` for new features
- `fix/bug-description` for bug fixes
- `docs/documentation-topic` for documentation changes
- `refactor/refactoring-description` for refactoring

### Commit Messages

We use semantic commit messages:

- `feat: add new distribution type`
- `fix: correct parameter ordering in histogram`
- `docs: update contributing guide`
- `test: add tests for composite distributions`
- `refactor: simplify parameter validation`
- `style: format code with ruff`
- `chore: update dependencies`

Commits should be:

- **Small and focused**: Each commit should represent a single logical change
- **Well-described**: Explain _why_ the change was made, not just _what_ changed
- **Frequent**: Commit often to track your progress

## Pull Request Process

1. **Create a WIP branch** for your work if you don't have an existing branch
2. **Make your changes** following the code standards
3. **Write tests** for your changes
4. **Run the test suite** to ensure everything passes:

   ```bash
   nox -s tests
   ```

5. **Run linting** to check code quality:

   ```bash
   pre-commit run --all-files
   ```

6. **Update documentation** if needed
7. **Commit your changes** with semantic commit messages
8. **Push to your fork** and create a pull request
9. **Respond to review feedback** promptly

### Pull Request Guidelines

- Provide a clear description of what your PR does
- Reference any related issues
- Include examples or use cases if applicable
- Ensure CI passes before requesting review
- Keep PRs focused on a single feature or fix

### Code Review

- Be patient and respectful during code review
- Address all review comments
- Push back with technical reasoning if you disagree
- Don't take criticism personally - we're all learning

## Where to Get Help

If you need help:

- Open a [discussion](https://github.com/scipp-atlas/pyhs3/discussions) for
  questions
- Check existing [issues](https://github.com/scipp-atlas/pyhs3/issues) for
  similar problems
- Read the [development](development.rst) guide for detailed workflow
  information
- Review the [architecture](architecture.rst) guide to understand the codebase
  structure

## Community Guidelines

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md). We are
committed to providing a welcoming and inclusive environment for all
contributors.

## Additional Resources

- [Testing Guide](testing.rst) - Comprehensive testing guide
- [Development Guide](development.rst) - Development workflow and tools
- [Architecture Overview](architecture.rst) - Codebase architecture overview
- [.github/CONTRIBUTING.md](https://github.com/scipp-atlas/pyhs3/blob/main/.github/CONTRIBUTING.md) -
  Quick reference guide
- [Scientific Python Developer Guide](https://learn.scientific-python.org/development/) -
  Detailed best practices
