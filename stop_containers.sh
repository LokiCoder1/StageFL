#!/bin/bash

NODES=$1  # Numero di nodi ricevuto come argomento
PROJECT_NAME="pytorch_project"  
T="client" 


for i in $(seq 1 $NODES); do
    
    PARTITION=$((i - 1))  # Calcola il valore di PARTITION
    
# Aggiunge il prefisso 0 se i < 10
    if [ $i -lt 11 ]; then
        NODE_NAME="fd-0$PARTITION"
    else
        NODE_NAME="fd-$PARTITION"
    fi

    echo "Arresto container su $NODE_NAME..."

    ssh -tt $NODE_NAME "docker stop ${PROJECT_NAME}_${T} && docker container prune -f"

done