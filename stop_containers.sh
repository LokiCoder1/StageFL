#!/bin/bash

NODES=$1  # Numero di nodi ricevuto come argomento
PROJECT_NAME="pytorch_project"  
T="client" 

for i in $(seq 1 $NODES); do
    # Aggiunge il prefisso 0 se i < 10
    TMP=$((i - 1))
    if [ $i -lt 11 ]; then
        NODE_NAME="fd-0$TMP"
    else
        NODE_NAME="fd-$TMP"
    fi

    echo "Arresto container su $NODE_NAME..."

    ssh $NODE_NAME <<EOF
        docker stop ${PROJECT_NAME}_${T}
        docker container prune -f
EOF

done