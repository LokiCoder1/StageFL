#!/bin/bash


###
# This script is used to deploy the C3S application on a SLURM cluster.
# It identifies the nodes allocated to the job and prints them out.
# The first node is designated as the coordinator, and the rest are worker nodes.
###
nodelist=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/NodeList|ExcNodeList/ {print $2}')
nodes=($(scontrol show hostnames "$nodelist"))

###
# Find the index of the first node that starts with "broadwell"
###
start_index=0
for i in "${!nodes[@]}"; do
    if [[ "${nodes[i]}" == broadwell* ]]; then
        start_index=$i
        break
    fi
done

nodes=("${nodes[@]:$start_index}")


###

i=0
for node in "${nodes[@]}"; do
    if [ $i -eq 0 ]; then
        echo -e "\n\n\n\nCoordinator: $node"
        echo -e "Setting up environment on $node\n\n"
        make build .
        i=$((i + 1))
        continue
    fi

    echo -e "\n\n\n\nWorking Node: $node"
    echo -e "Setting up environment on $node\n\n"
    ssh "$node" "cd Stage/StageFL/ && \
    make build ."
done


