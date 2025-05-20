#!/bin/bash
source /cpfs01/shared/mascience/lvhaoming/workspace/fankaixuan/EasyR1/examples/env.sh

dlcrun \
    -n test_multi_nodes \
    -g $GPUS_PER_NODE \
    -j $NNODES \
    /cpfs01/shared/mascience/lvhaoming/workspace/fankaixuan/EasyR1/examples/run_qwen2_5_vl_7b_bs.sh