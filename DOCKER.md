# Docker Deployment Guide for Hostinger VPS with Coolify

This guide explains how to deploy the RAG System using Coolify on a Hostinger VPS.

## 🚀 Coolify Deployment (Recommended)

1.  **Add a Resource in Coolify**:
    *   Select **"Private GitHub Repository"** (or Public).
    *   Choose your repository: `rag-system`.
    *   **Build Pack**: Select **Docker Compose**.

2.  **Configuration**:
    *   Coolify will automatically detect the `docker-compose.yml`.
    *   Ensure the **Domains** section is configured (e.g., `https://api.yourdomain.com`).
    *   Port 8080 is exposed by default.

3.  **Environment Variables**:
    *   Go to the **Environment Variables** tab in Coolify.
    *   Copy the contents of `.env.coolify.example`.
    *   Fill in your actual production values.

4.  **Deploy**:
    *   Click **Deploy**.
    *   Coolify will build the Docker image using the provided `Dockerfile`.

## Prerequisites (Local or Manual)

- Docker installed (version 20.10 or later)
- Docker Compose installed (version 2.0 or later)
- A `.env` file with required environment variables (see below)

## Quick Start

### 1. Create Environment File

Create a `.env` file in the project root with the following variables:

```bash
# Flask Configuration
PORT=8080
FLASK_ENV=production

# Supabase Configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
SUPABASE_DATABASE_PASSWORD=your_database_password_here
DB_CONNECTION2=postgresql://user:password@host:port/database

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_KEY=your_openai_api_key_here

# DeepSeek Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# LlamaParse Configuration
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here

# Linkup API (Optional)
LINKUP_API_KEY=your_linkup_api_key_here
```

### 2. Build and Run with Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### 3. Build and Run with Docker

```bash
# Build the image
docker build -t rag-system .

# Run the container
docker run -d \
  --name rag-system \
  -p 8080:8080 \
  --env-file .env \
  rag-system

# View logs
docker logs -f rag-system

# Stop the container
docker stop rag-system
docker rm rag-system
```

## Accessing the Application

Once the container is running, the application will be available at:

- **API Base URL**: `http://localhost:8080`
- **Health Check**: `http://localhost:8080/health`

## Health Check

The application includes a health check endpoint at `/health` that returns:

```json
{
  "status": "healthy",
  "service": "rag-system",
  "timestamp": "2024-01-01T12:00:00"
}
```

## Environment Variables

### Required Variables

- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_KEY` - Your Supabase API key
- `SUPABASE_DATABASE_PASSWORD` - Database password
- `OPENAI_API_KEY` - OpenAI API key
- `DEEPSEEK_API_KEY` - DeepSeek API key
- `LLAMA_CLOUD_API_KEY` - LlamaParse API key

### Optional Variables

- `PORT` - Port to run the application on (default: 8080)
- `LINKUP_API_KEY` - Linkup API key for enhanced research
- `DB_CONNECTION2` - Alternative database connection string

## Docker Compose Configuration

The `docker-compose.yml` file includes:

- **Port mapping**: Maps container port 8080 to host port 8080 (configurable via `PORT` env var)
- **Volume mounting**: Mounts `./uploads` directory for file uploads
- **Health checks**: Automatic health monitoring
- **Restart policy**: Automatically restarts on failure
- **Network**: Isolated Docker network for the application

## Building for Production

For production deployments, consider:

1. **Multi-stage builds** to reduce image size
2. **Non-root user** for security
3. **Resource limits** in docker-compose.yml
4. **Secrets management** (Docker secrets, AWS Secrets Manager, etc.)

Example production docker-compose.yml additions:

```yaml
services:
  rag-system:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    user: "1000:1000"  # Non-root user
```

## Troubleshooting

### Container won't start

1. Check logs: `docker-compose logs rag-system`
2. Verify environment variables are set correctly
3. Ensure port 8080 is not already in use

### Health check failing

1. Check if the application is running: `docker exec rag-system ps aux`
2. Test the health endpoint manually: `curl http://localhost:8080/health`
3. Review application logs for errors

### Out of memory errors

1. Increase Docker memory limits
2. Reduce `MAX_CONTENT_LENGTH` in the application
3. Monitor memory usage: `docker stats rag-system`

## Updating the Application

```bash
# Pull latest code
git pull

# Rebuild the image
docker-compose build

# Restart the container
docker-compose up -d
```

## Development Mode

For development with live code reloading:

```bash
# Mount source code as volume
docker run -d \
  --name rag-system-dev \
  -p 8080:8080 \
  --env-file .env \
  -v $(pwd):/app \
  rag-system
```

Note: This requires the application to support hot-reloading (Flask debug mode).

## Security Considerations

1. **Never commit `.env` files** to version control
2. **Use Docker secrets** for sensitive data in production
3. **Run as non-root user** in production
4. **Keep base images updated** regularly
5. **Scan images** for vulnerabilities: `docker scan rag-system`

## Support

For issues or questions, check:
- Application logs: `docker-compose logs -f`
- Container status: `docker-compose ps`
- Health endpoint: `curl http://localhost:8080/health`

