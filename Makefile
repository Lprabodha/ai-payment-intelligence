DOCKER_COMPOSE_FILE=docker-compose.yml

# Default target
.PHONY: all
all: build install up

# Build the Docker image
.PHONY: build
build:
	docker compose -f $(DOCKER_COMPOSE_FILE) build

# Install the requirements
.PHONY: install
install:
	@docker exec -it ai-payment-intelligence-app /bin/sh -c "pip install --upgrade pip && pip install -r requirements.txt"

# Start the Docker Compose services
.PHONY: up
up:
	docker compose -f $(DOCKER_COMPOSE_FILE) up -d

# Stop the Docker Compose services
.PHONY: down
down:
	docker compose -f $(DOCKER_COMPOSE_FILE) down

# Run a Python command inside the Docker container
.PHONY: run
run:
	docker compose -f $(DOCKER_COMPOSE_FILE) exec app python /src/ai_models/fraud_detection.py

.PHONY: shell
shell:
	docker compose -f $(DOCKER_COMPOSE_FILE) exec app /bin/bash

.PHONY: clear-cache
clear-cache:
	docker container prune -f
	docker image prune -a -f
	docker volume prune -f
	docker network prune -f
	docker builder prune -a -f