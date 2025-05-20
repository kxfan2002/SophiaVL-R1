set -x

source /cpfs04/shared/ai4phys/share/fankaixuan/EasyR1/examples/env_single.sh

export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=47a69bdaf6eae25916e84d3e036bda9d9fd8446f
OUTPUT_DIR=/cpfs04/shared/ai4phys/share/fankaixuan/EasyR1/outputs/qwen2vl_7b_sft_fullsets
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
export HF_HUB_CACHE="/cpfs04/shared/ai4phys/share/huggingface_hub"
export HF_ENDPOINT=https://hf-mirror.com
# MODEL_PATH=/oss/ai4chem/chemshare/workspace/mascience/model/chemr1/sft/full/2025_2_26r_qwen2.5_vl_Instruct/
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer. \n
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE enclosed within <answer> </answer> tags. If you don't follow this format, your reward will be low. If you use formula, please use LaTeX format."""

python -m verl.trainer.main \
    config=examples/0329/fullsets.yaml \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.max_prompt_length=8192 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_sft_fullsets \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log_032902_fullsets.txt" 

