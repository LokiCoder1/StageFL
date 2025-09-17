# container Docker management

PROJECT_NAME = pytorch_project
IMAGE_NAME = fd-coordinator:5000/pytorch_image:latest

.PHONY: start build run stop clean shell ssh train 1

setup:
	@bash env_setup.sh $(NODES)

# start container on both server and client
# arg 1: 
start:
	@if [ -z "$(NODES)"]; then \
		echo "Please provide the number of nodes to start"; \
		exit 1; \
	fi
	@echo "Starting container $(PROJECT_NAME)_server"
	$(MAKE) run T=server

	@sleep 15
# @docker exec -it pytorch_project_server pip install wandb
	@echo "Starting containers $(PROJECT_NAME)_client"
	$(MAKE) ssh NODES=$(NODES) 

	@echo "Start training..."
	$(MAKE) train



# to Build Docker   image
build:
	docker build -t $(IMAGE_NAME) .

# background Docker container run
# T=server for server container activation
# T=client for client container activation (it will need further variables)
run:
ifeq ($(T),server)

	@docker run -d --name $(PROJECT_NAME)_server -v $(shell pwd):/app --network=host --rm $(IMAGE_NAME) sh -c "flower-superlink --insecure 2>&1 | tee server_connections.log"
endif

#SUPERLINK -> server to connect with
#PARTITION -> node in progressive order
#NUM_PARTITIONS -> total nodes number
ifeq ($(T),client)

	@docker run -d --name $(PROJECT_NAME)_client -v $(shell pwd):/app --network=host --rm $(IMAGE_NAME) sh -c 'flower-supernode --insecure --superlink fd-coordinator:9092 --node-config "partition-id=$(PARTITION) num-partitions=$(NUM_PARTITIONS)" --max-retries 30 --max-wait-time 600.0 2>&1 | tee client_output_$(PARTITION).log'
endif


# stops and remove container 
# NODES -> number of active nodes to be stopped
stop:
	@if [ -z "$(NODES)"]; then \
		echo "Please provide the number of nodes to stop"; \
		exit 1; \
	fi
	@echo "Stopping container $(PROJECT_NAME)_server"
	@docker stop $(PROJECT_NAME)_server; 
	@docker container prune -f; 
	@echo "Stopping containers $(PROJECT_NAME)_client" 
	@bash stop_containers.sh $(NODES)


# clean Docker image
clean: stop
	docker rmi $(IMAGE_NAME)


#automatization in creating containers in different vm 
#NODES -> number of nodes to activate
ssh: 
	@bash run_nodes.sh $(NODES)
	@bash wandb_setup.sh $(NODES)


#start training
train:
	@ docker exec -it $(PROJECT_NAME)_server sh -c "cd /app/pytorchtest && flwr run"  2>&1 | tee $(T)_output.log
	


# open container shell 
#T -> server / client 
shell:
	docker exec -it $(PROJECT_NAME)_$(T) /bin/bash


1:
	docker exec -it $(PROJECT_NAME)_server sh -c "pip install wandb"
#	@docker exec -it $(PROJECT_NAME)_server sh -c "wandb login --relogin"