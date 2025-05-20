set -x

source /cpfs04/shared/ai4phys/share/fankaixuan/EasyR1/examples/env.sh

export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=47a69bdaf6eae25916e84d3e036bda9d9fd8446f
OUTPUT_DIR=/cpfs04/shared/ai4phys/share/fankaixuan/EasyR1/outputs/qwen2vl_7b_mn
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
ls -ld "$OUTPUT_DIR"
export HF_HUB_CACHE="/cpfs04/shared/ai4phys/share/huggingface_hub"
export HF_ENDPOINT=https://hf-mirror.com
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct   # replace it with your local file path

if [ "$RANK" -eq 0 ]; then
  echo "MASTER_PORT: $MASTER_PORT"
  ray start --head --port=$MASTER_PORT --dashboard-host=0.0.0.0
  sleep 90
  echo "Executing main program on head node..."
  ray status
  ray job submit --address="http://127.0.0.1:8265" \
    -- python -m verl.trainer.main \
    config=examples/grpo_bs.yaml \
    data.max_prompt_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_bs \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log_031302.txt" 
else
  sleep 30
  ray start --address=$MASTER_ADDR:$MASTER_PORT
  echo "Start worker node (RANK=${RANK})"
  sleep 120
  while true; do
    status=$(ray status 2>&1)
    if echo "$status" | grep -q "Active: "; then
      echo "worker nodes still active"
      sleep 600
    else
      echo "Exiting..."
      exit 0
    fi
  done
fi