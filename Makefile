# container Docker management

PROJECT_NAME = pytorch_project
IMAGE_NAME = fd-coordinator:5000/pytorch_image

.PHONY: start build run stop clean shell ssh train 1

# start container on both server and client
# arg 1: 
start:
	@echo "Starting container $(PROJECT_NAME)_server"
	$(MAKE) run T=server
	@echo "Starting containers $(PROJECT_NAME)_client"
	$(MAKE) ssh NODES=$(NODES) 



# to Build Docker image: DEPRECATED
build:
	docker build -t $(IMAGE_NAME) .

# background Docker container run
# T=server for server container activation
# T=client for client container activation (it will need further variables)
run:
ifeq ($(T),server)
# test code -> not working
#	docker run --rm -it --network=host -v $(shell pwd):/app --name $(PROJECT_NAME)_server -d flwr_serverapp:0.0.1 sh -c 'flower-superlink --insecure --serverappio-api-address fd-coordinator:9091 2>&1 | tee server_connections.log'
#prev code	
	@docker run -d --name $(PROJECT_NAME)_server -v $(shell pwd):/app --network=host --rm $(IMAGE_NAME) sh -c "flower-superlink --insecure 2>&1 | tee server_connections.log"
endif

#SUPERLINK -> server to connect with
#PARTITION -> node in progressive order
#NUM_PARTITIONS -> total nodes number
ifeq ($(T),client)
# test code -> not working
#	docker run --rm -it --network host --name $(PROJECT_NAME)_client -v $(shell pwd):/app -d $(IMAGE_NAME) sh -c 'flower-supernode --insecure -superlink fd-coordinator:9092 --node-config "partition=$(PARTITION) num-partitions=$(NUM_PARTITIONS)" --clientappio-api-address supernode-1:9094 2>&1 |tee client_output_$(PARTITION).log'
#prev code
	docker run -d --name $(PROJECT_NAME)_client -v $(shell pwd):/app --network=host --rm $(IMAGE_NAME) sh -c 'flower-supernode --insecure --superlink fd-coordinator:9092 --node-config "partition-id=$(PARTITION) num-partitions=$(NUM_PARTITIONS)" 2>&1 | tee client_output_$(PARTITION).log'
endif


# stops and remove container 
# NODES -> number of active nodes to be stopped
stop:
	@if [ -z "$(NODES)"]; then \
		docker stop $(PROJECT_NAME)_server; \
		docker container prune -f; \
		exit 1; \
	fi	
	@bash stop_containers.sh $(NODES)


# clean Docker image
clean: stop
	docker rmi $(IMAGE_NAME)


#automatization in creating containers in different vm 
#NODES -> number of nodes to activate
ssh: 
	@bash run_nodes.sh $(NODES)


#start training
train:
	docker exec -it $(PROJECT_NAME)_server sh -c "cd /app/pytorchtest && flwr run" 2>&1 | tee $(T)_output.log
	
#TODO: fix connection between server start and client start


# open container shell 
#T -> server / client 
shell:
	docker exec -it $(PROJECT_NAME)_$(T) /bin/bash


1: