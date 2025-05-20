NODES=$1  # Numero di nodi ricevuto come argomento
WANDB_API_KEY="c9ecc4c3eeac8445768b6c97a55298ddd835562d"  

for i in $(seq 1 $NODES); do
    
    PARTITION=$((i - 1))  # Calcola il valore di PARTITION
    
# Aggiunge il prefisso 0 se i < 10
    if [ $i -lt 11 ]; then
        NODE_NAME="fd-0$PARTITION"
    else
        NODE_NAME="fd-$PARTITION"
    fi
    echo "Connessione a wandb..."
    ssh -tt $NODE_NAME  "docker exec -it pytorch_project_client sh -c \"wandb login $WANDB_API_KEY\""
done
