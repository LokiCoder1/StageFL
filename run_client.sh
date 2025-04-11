#!/bin/bash

cd pytorchtest/

echo "flower-supernode --insecure --superlink fd-coordinator:9092 --node-config "partition=$1 num-partitions=$2" 2>&1 | tee client_output_$1.log"

flower-supernode --insecure --superlink fd-coordinator:9092 --node-config "partition=${1} num-partitions=${2}" 2>&1 | tee client_output_$1.log
