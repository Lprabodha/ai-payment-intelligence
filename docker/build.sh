#!/bin/bash

# Docker build script with caching optimizations
set -e

IMAGE_NAME="ai-payment-intelligence"
CONTAINER_NAME="ai-payment-intelligence"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to enable BuildKit
enable_buildkit() {
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    print_status "BuildKit enabled for better caching"
}

# Function to build with cache optimization
build_with_cache() {
    local dockerfile=${1:-"docker/Dockerfile"}
    local tag=${2:-$IMAGE_NAME}
    
    print_status "Building image with cache optimization..."
    
    # Enable BuildKit
    enable_buildkit
    
    # Build with cache from previous image
    docker build \
        --cache-from $tag:latest \
        --tag $tag \
        --file $dockerfile \
        .
    
    print_status "Build completed successfully!"
}

# Function to build multi-stage (now default)
build_multistage() {
    print_status "Building multi-stage image..."
    build_with_cache "docker/Dockerfile" "${IMAGE_NAME}:latest"
}

# Function to build without cache
build_no_cache() {
    print_status "Building without cache (clean build)..."
    
    enable_buildkit
    
    docker build \
        --no-cache \
        --tag $IMAGE_NAME \
        --file docker/Dockerfile \
        .
    
    print_status "Clean build completed!"
}

# Function to run container
run_container() {
    print_status "Starting container..."
    
    # Stop and remove existing container if it exists
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    
    # Run new container
    docker run -d \
        --name $CONTAINER_NAME \
        --restart=always \
        -v $(pwd)/src/data/models:/src/data/models \
        -v $(pwd)/src/data/raw:/src/data/raw \
        -p 8010:8010 \
        $IMAGE_NAME
    
    print_status "Container started successfully!"
    print_status "Application available at: http://localhost:8010"
}

# Function to show build cache info
show_cache_info() {
    print_status "Docker build cache information:"
    docker system df
    echo ""
    print_status "Image layers:"
    docker history $IMAGE_NAME 2>/dev/null || print_warning "Image not found"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker system prune -f
    docker image prune -f
    print_status "Cleanup completed!"
}

# Function to rebuild
rebuild() {
    print_status "Rebuilding container..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    build_with_cache
    run_container
}

# Function to show logs
show_logs() {
    docker logs -f $CONTAINER_NAME
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build image with cache optimization"
    echo "  multistage  Build multi-stage image (now default)"
    echo "  no-cache    Build without cache (clean build)"
    echo "  run         Run container"
    echo "  rebuild     Rebuild and run container"
    echo "  logs        Show container logs"
    echo "  cache       Show cache information"
    echo "  clean       Clean up Docker resources"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build && $0 run"
    echo "  $0 rebuild"
    echo "  $0 optimized"
}

# Main script logic
main() {
    check_docker
    
    case "${1:-help}" in
        "build")
            build_with_cache
            ;;
        "multistage")
            build_multistage
            ;;
        "no-cache")
            build_no_cache
            ;;
        "run")
            run_container
            ;;
        "rebuild")
            rebuild
            ;;
        "logs")
            show_logs
            ;;
        "cache")
            show_cache_info
            ;;
        "clean")
            cleanup
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"
