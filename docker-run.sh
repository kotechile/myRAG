#!/bin/bash

# Docker helper script for RAG System
# Makes it easy to build, run, and manage the Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  Warning: .env file not found${NC}"
    echo "Please create a .env file with required environment variables."
    echo "See DOCKER.md for details."
    exit 1
fi

# Function to build the image
build() {
    echo -e "${GREEN}🔨 Building Docker image...${NC}"
    docker build -t rag-system .
    echo -e "${GREEN}✅ Build complete!${NC}"
}

# Function to run with docker-compose
run_compose() {
    echo -e "${GREEN}🚀 Starting with Docker Compose...${NC}"
    docker-compose up -d
    echo -e "${GREEN}✅ Container started!${NC}"
    echo "View logs with: docker-compose logs -f"
    echo "Stop with: docker-compose down"
}

# Function to run with docker
run_docker() {
    echo -e "${GREEN}🚀 Starting with Docker...${NC}"
    docker run -d \
        --name rag-system \
        -p 8080:8080 \
        --env-file .env \
        rag-system
    echo -e "${GREEN}✅ Container started!${NC}"
    echo "View logs with: docker logs -f rag-system"
    echo "Stop with: docker stop rag-system"
}

# Function to stop
stop() {
    if docker ps -a | grep -q rag-system; then
        echo -e "${YELLOW}🛑 Stopping container...${NC}"
        docker stop rag-system 2>/dev/null || docker-compose down
        echo -e "${GREEN}✅ Container stopped!${NC}"
    else
        echo -e "${YELLOW}No running container found${NC}"
    fi
}

# Function to view logs
logs() {
    if docker ps -a | grep -q rag-system; then
        docker logs -f rag-system
    else
        docker-compose logs -f
    fi
}

# Function to check status
status() {
    echo -e "${GREEN}📊 Container Status:${NC}"
    docker ps -a | grep rag-system || echo "No container found"
    echo ""
    echo -e "${GREEN}🏥 Health Check:${NC}"
    curl -s http://localhost:8080/health | python -m json.tool 2>/dev/null || echo "Service not responding"
}

# Function to clean up
clean() {
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
    docker stop rag-system 2>/dev/null || true
    docker rm rag-system 2>/dev/null || true
    docker-compose down 2>/dev/null || true
    echo -e "${GREEN}✅ Cleanup complete!${NC}"
}

# Function to rebuild
rebuild() {
    clean
    build
    run_compose
}

# Main command handler
case "${1:-help}" in
    build)
        build
        ;;
    run|start)
        if command -v docker-compose &> /dev/null; then
            run_compose
        else
            run_docker
        fi
        ;;
    stop)
        stop
        ;;
    logs)
        logs
        ;;
    status|health)
        status
        ;;
    clean|rm)
        clean
        ;;
    rebuild)
        rebuild
        ;;
    help|--help|-h)
        echo "Docker Helper Script for RAG System"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  build     - Build the Docker image"
        echo "  run       - Start the container (uses docker-compose if available)"
        echo "  stop      - Stop the container"
        echo "  logs      - View container logs"
        echo "  status    - Check container and health status"
        echo "  clean     - Stop and remove containers"
        echo "  rebuild   - Clean, build, and start"
        echo "  help      - Show this help message"
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac

