# Dockerfile for RAG System
# Uses Python 3.13 for compatibility with all dependencies

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Expose port 8080 for standard web traffic (non-root)
EXPOSE 8080

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN addgroup --system appgroup && adduser --system --group appuser

# Copy application code and set permissions
COPY . .
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Run the application with Gunicorn on port 8080
# Using create_app() factory pattern
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--access-logfile", "-", "--workers", "2", "--threads", "4", "--timeout", "120", "main:create_app()"]

