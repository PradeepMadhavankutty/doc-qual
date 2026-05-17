# Contributing to Doc-Qual

Thank you for your interest in contributing! This document describes how to get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/PradeepMadhavankutty/doc-qual.git
cd doc-qual

# Create a virtual environment and install in editable mode
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest --tb=short -q
```

## Code Style

This project uses **ruff**, **black**, and **isort** for formatting and linting:

```bash
ruff check .          # lint
black . --fast        # format
isort .               # import order
mypy doc_qual/        # type checking
```

All checks must pass before a pull request is merged.

## Pull Request Process

1. Fork the repository and create a branch from `main`.
2. Write tests for any new feature or bug fix.
3. Run the full quality gate (`ruff`, `black`, `pytest`) locally.
4. Open a pull request with a clear description of the change.
5. Address reviewer feedback.

## Reporting Issues

Use [GitHub Issues](https://github.com/PradeepMadhavankutty/doc-qual/issues) to report bugs or request features. Please include:
- Python version
- Operating system
- Minimal reproduction case
- Expected vs. actual behaviour

## Licence

By contributing you agree that your contributions will be licenced under the MIT Licence.
