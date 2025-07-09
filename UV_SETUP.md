# UV Package Manager Setup

This project now uses [uv](https://github.com/astral-sh/uv) as the package manager for faster and more reliable dependency management.

## Simplified Dependencies

The project has been streamlined to use only essential dependencies:

```txt
pydantic-ai==0.4.0   # Core AI framework
fastapi==0.115.12    # Web framework
logfire-api==3.16.1  # Logging
uvicorn>=0.34.0      # ASGI server
tavily-python==0.5.1 # tavily
```

This results in **103 total packages** (including transitive dependencies), down from the original 191+ packages.

## Quick Start

```bash
# Install dependencies and create virtual environment
uv sync

# Run the application
uv run uvicorn src.cortana.Cortana_api:app --host 0.0.0.0 --port 8000

# Run any Python command in the virtual environment
uv run python -c "import pydantic_ai, fastapi; print('Dependencies work!')"

# Install development dependencies
uv sync --group dev

# Add a new dependency
uv add package-name

# Remove a dependency
uv remove package-name

# Update dependencies
uv lock --upgrade
uv sync

# Check lock file consistency
uv lock --check
```

## To Skip Virtual Environment Creation

If you don't want uv to create a `.venv` directory:

```bash
# Install to system Python
uv sync --system

# Or use an existing virtual environment
source your-venv/bin/activate  # Linux/Mac
your-venv\Scripts\activate     # Windows
uv sync --no-install-project
```

## Key Files

- `pyproject.toml` - Project configuration and dependencies (Hide the readme part for hugging face spaces)
- `uv.lock` - Locked dependency versions (132KB, down from 334KB)
- `requirements.txt` - Simplified fallback dependencies (4 lines)
- `.venv/` - Virtual environment (auto-created by uv)

## Benefits of UV

- **Speed**: Up to 10-100x faster than pip
- **Reliability**: Consistent dependency resolution
- **Lean**: Minimal dependencies for faster installs
- **Lock files**: Reproducible builds across environments
- **Cache**: Shared dependency cache across projects

## Migration Notes

- Dependencies reduced from 170+ to 4 core packages
- Total installed packages: 103 (down from 191)
- Lock file size: 132KB (down from 334KB)
- Faster installs and smaller Docker images
- Development dependencies managed through `[tool.uv]` section in pyproject.toml 