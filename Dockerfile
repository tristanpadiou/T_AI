# Use Python 3.13.2 as the base image
FROM python:3.13.2-slim

# Install system dependencies as root first
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create user after system packages are installed
RUN useradd -m -u 1000 user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Create directories with proper permissions for app data
RUN mkdir -p $HOME/app/logs $HOME/app/data $HOME/app/tmp && \
    chmod -R 755 $HOME/app

# Copy pyproject.toml and uv.lock first to leverage Docker cache
COPY --chown=user:user pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN uv sync --frozen --no-dev

# Copy the application code
COPY --chown=user:user . $HOME/app

# Ensure all app files have proper permissions
RUN chmod -R 755 $HOME/app && \
    find $HOME/app -type f -name "*.py" -exec chmod 644 {} \;

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application using uv
CMD ["uv", "run", "uvicorn", "src.cortana.Cortana_api:app", "--host", "0.0.0.0", "--port", "8000"] 