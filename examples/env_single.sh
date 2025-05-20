# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/cpfs04/shared/ai4phys/fankaixuan/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cpfs01/shared/mascience/lvhaoming/workspace/share/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/cpfs01/shared/mascience/lvhaoming/workspace/share/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/cpfs04/shared/ai4phys/fankaixuan/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate easyr1

GPUS=8
GPUS_PER_NODE=8
NNODES=$(expr $GPUS / $GPUS_PER_NODE)
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-28597}
