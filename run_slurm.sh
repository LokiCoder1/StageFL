#!/bin/bash
#SBATCH --job-name=federated-learning
#SBATCH --time=00:30:00
#SBATCH --nodes=4
#SBATCH --partition=broadwell
#SBATCH --exclusive
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1

SHARED_DIR="$HOME/Stage/StageFL"
NUM_CLIENTS=3

# Ottieni la lista dei nodi
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
SERVER_NODE=${NODES[0]}

echo "Nodes allocated: ${NODES[@]}"
echo "Server node: $SERVER_NODE"

# Avvia il server FL su nodo specifico
echo "Starting FL server on $SERVER_NODE..."
srun --nodes=1 --ntasks=1 --nodelist=$SERVER_NODE \
    bash -c "cd $SHARED_DIR && occam-run -n $SERVER_NODE damianocann/pytorch-flower-app flower-superlink --insecure" &

sleep 20

# Avvia i client FL
for i in $(seq 0 $((NUM_CLIENTS-1))); do
    CLIENT_NODE=${NODES[$((i+1))]}
    echo "Starting client $i on $CLIENT_NODE..."
    srun --nodes=1 --ntasks=1 --nodelist=$CLIENT_NODE \
        bash -c "cd $SHARED_DIR && occam-run -n $CLIENT_NODE damianocann/pytorch-flower-app flower-supernode --insecure --superlink $SERVER_NODE:9092 --node-config 'partition-id=$i num-partitions=$NUM_CLIENTS'" &
done

# Attendi che tutti siano connessi
sleep 30

# Avvia il training
echo "Starting training on $SERVER_NODE..."
srun --nodes=1 --ntasks=1 --nodelist=$SERVER_NODE \
    bash -c "cd $SHARED_DIR && occam-run -n $SERVER_NODE damianocann/pytorch-flower-app sh -c 'cd /app/pytorchtest && flwr run'"

echo "Federated Learning completed!"