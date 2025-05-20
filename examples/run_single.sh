#!/bin/bash
source /cpfs04/shared/ai4phys/share/fankaixuan/EasyR1/examples/env_single.sh

dlcrun \
    -n single_node \
    -g $GPUS_PER_NODE \
    -j $NNODES \
    -w ai4chem \
    -p 6 \
    /cpfs04/shared/ai4phys/share/fankaixuan/EasyR1/examples/run_qwen2_5_vl_7b_geo.sh