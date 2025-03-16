.PHONY: build run shell stop rm logs

IMAGE_NAME=ai-payment
CONTAINER_NAME=ai-payment

build:
	docker build -t $(IMAGE_NAME) -f docker/Dockerfile .

up:
	docker run -d --name $(CONTAINER_NAME) --restart=always -p 8010:8010 $(IMAGE_NAME)

down:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

logs:
	docker logs -f $(CONTAINER_NAME)