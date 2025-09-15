NODES=$1  # Numero di nodi ricevuto come argomento

echo "Rebuilding $NODES nodes. Starting..."


for i in $(seq 1 $NODES); do
    
    PARTITION=$((i - 1))  # Calcola il valore di PARTITION
    
# Aggiunge il prefisso 0 se i < 10
    if [ $i -lt 11 ]; then
        NODE_NAME="fd-0$PARTITION"
    else
        NODE_NAME="fd-$PARTITION"
    fi

    echo "Setup container in $NODE_NAME..."
    ssh -tt $NODE_NAME "docker system prune -af && docker volume prune -f"
    
    ssh -tt $NODE_NAME  "cd pytorchtest/ && \
    make build"

    ssh -tt $NODE_NAME "cd pytorchtest/ && \
    make run T=client SUPERLINK=fd-coordinator:9092 PARTITION=$PARTITION NUM_PARTITIONS=$TOT_NODES && \
    docker exec -it pytorch_project_client pip show wandb || \
    docker exec -it pytorch_project_client pip install wandb"

done

echo "All nodes have been rebuilt."
ssh -tt $NODE_NAME "cd pytorchtest && \
    make stop NODES=$NODES"