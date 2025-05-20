#!/bin/bash
source /cpfs01/shared/mascience/lvhaoming/workspace/fankaixuan/EasyR1/examples/env_single.sh

dlcrun \
    -n test_multi_nodes \
    -g $GPUS_PER_NODE \
    -j $NNODES \
    -q C \
    -p 6 \
    echo "DLC demo"