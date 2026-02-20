# Contributing to generic-robot-env

Thank you for your interest in contributing to `generic-robot-env`!

## Code of Conduct

Please be respectful and professional in all interactions.

## How to Contribute

1.  **Report Bugs**: Open an issue on GitHub describing the bug and how to reproduce it.
2.  **Suggest Features**: Open an issue to discuss new features.
3.  **Pull Requests**:
    *   Fork the repository.
    *   Create a new branch for your changes.
    *   Add tests for any new functionality.
    *   Ensure all tests pass and linting is clean.
    *   Submit a PR.

## Development Setup

This project uses `uv` for dependency management.

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --all-extras
```

### Testing

Run tests with `pytest`:

```bash
uv run pytest
```

### Linting

We use `ruff` for linting and formatting.

```bash
uv run ruff check .
uv run ruff format .
```
