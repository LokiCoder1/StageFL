#!/bin/bash

NODES=$1  # Numero di nodi ricevuto come argomento

for i in $(seq 1 $NODES); do
    
    PARTITION=$((i - 1))  # Calcola il valore di PARTITION
    
# Aggiunge il prefisso 0 se i < 10
    if [ $i -lt 11 ]; then
        NODE_NAME="fd-0$PARTITION"
    else
        NODE_NAME="fd-$PARTITION"
    fi

    TOT_NODES=$NODES
    echo "Connessione a $NODE_NAME...; numero nodi = $TOT_NODES"
    
    ssh -tt $NODE_NAME "cd pytorchtest/ && \
        make run T=client SUPERLINK=fd-coordinator:9092 PARTITION=$PARTITION NUM_PARTITIONS=$TOT_NODES"

done