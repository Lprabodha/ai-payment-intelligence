.PHONY: build build-optimized build-no-cache up down shell logs clean

IMAGE_NAME=ai-payment-intelligence
CONTAINER_NAME=ai-payment-intelligence

# Standard build with caching
build:
	docker build -t $(IMAGE_NAME) -f docker/Dockerfile .

# Multi-stage build (now default)
build-multistage:
	docker build -t $(IMAGE_NAME) -f docker/Dockerfile .

# Build without cache (for debugging)
build-no-cache:
	docker build --no-cache -t $(IMAGE_NAME) -f docker/Dockerfile .

# Build with BuildKit for better caching
build-buildkit:
	DOCKER_BUILDKIT=1 docker build -t $(IMAGE_NAME) -f docker/Dockerfile .

# Build with specific cache mount
build-cache:
	docker build --cache-from $(IMAGE_NAME) -t $(IMAGE_NAME) -f docker/Dockerfile .

up:
	docker run -d --name $(CONTAINER_NAME) --restart=always \
		-v $(PWD)/src/data/models:/src/data/models \
		-v $(PWD)/src/data/raw:/src/data/raw \
		-p 8010:8010 $(IMAGE_NAME)

down:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

logs:
	docker logs -f $(CONTAINER_NAME)

# Clean up Docker resources
clean:
	docker system prune -f
	docker image prune -f

# Rebuild and restart
rebuild: down build up

# Quick development cycle
dev: build up logs

# Docker Compose commands
compose-up:
	docker-compose -f docker/docker-compose.yml up --build

compose-down:
	docker-compose -f docker/docker-compose.yml down

compose-dev:
	docker-compose -f docker/docker-compose.dev.yml up --build

compose-dev-down:
	docker-compose -f docker/docker-compose.dev.yml down
