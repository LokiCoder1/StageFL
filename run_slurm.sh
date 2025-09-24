#!/bin/bash
#SBATCH --job-name=federated-learning
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1

# Configurazione
SHARED_DIR="~/Stage/StageFL"  
IMAGE="docker://damianocann/pytorch-flower-app:latest"
SERVER_NODE=$SLURM_NODELIST[0]  # Primo nodo come server
NUM_CLIENTS=3

# Avvia il server sul primo nodo
echo "Starting FL server on $SERVER_NODE..."
srun --ntasks 1 --nodes 1 -w $SERVER_NODE apptainer exec \
    --bind $SHARED_DIR:/app \
    --bind $SHARED_DIR/pytorchtest:/app/pytorchtest \
    $IMAGE \
    flower-superlink --insecure &

# Attendi che il server sia pronto
sleep 15

# Avvia i client sugli altri nodi
for i in $(seq 0 $((NUM_CLIENTS-1))); do
    NODE_IDX=$((i+1))
    CLIENT_NODE=$SLURM_NODELIST[$NODE_IDX]
    
    echo "Starting client $i on $CLIENT_NODE..."
    srun --ntasks 1 --nodes 1 -w $CLIENT_NODE apptainer exec \
        --bind $SHARED_DIR:/app \
        --bind $SHARED_DIR/pytorchtest:/app/pytorchtest \
        $IMAGE \
        flower-supernode --insecure --superlink $SERVER_NODE:9092 \
        --node-config "partition-id=$i num-partitions=$NUM_CLIENTS" &
done

# Attendi che tutti siano pronti
sleep 30

# Avvia il training
echo "Starting training..."
srun --ntasks 1 --nodes 1 -w $SERVER_NODE apptainer exec \
    --bind $SHARED_DIR:/app \
    $IMAGE \
    sh -c "cd /app/pytorchtest && flwr run"

# Aspetta che tutti i job finiscano
wait

echo "Federated Learning completed!"