# set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=47a69bdaf6eae25916e84d3e036bda9d9fd8446f
# export HF_HUB_CACHE="/cpfs04/shared/ai4phys/share/huggingface_hub"
export HF_ENDPOINT=https://hf-mirror.com
export OPENAI_API_URL="https://scvvnto1emaedi4plgu0g.apigateway-cn-beijing-inner.volceapi.com/mlp/s-20250414155823-v8qvc/v1/chat/completions"
# export OPENAI_API_KEY="sk-eZXm9L9HTBeAtwCGkKNNql6sgclAk90ofhTvwbuTGAWPtefT"
# export OPENAI_API_URL="https://sd00cb0l8e6v9i08l65o0.apigateway-cn-beijing-inner.volceapi.com/mlp/s-20250417170323-n8m78/v1/chat/completions"

export OPENAI_API_KEY="EMPTY"
export REWARD_MODEL="Qwen2.5-VL-3B-Instruct"
# export REWARD_MODEL="Qwen2.5-VL-3B-Instruct-origin"
export TRANSFORMERS_CACHE=/fs-computility/ai4phys/shared/checkpoints/

OUTPUT_DIR=/fs-computility/ai4chem/shared/r1_ckpt/full-op-27
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tagsdd. The final answer MUST BE enclosed within <answer> </answer> tags, for example <think>your_thinking_process</think><answer>your_final_answer</answer>. If you use formula, please use LaTeX format."""
MODEL_PATH=/fs-computility/ai4phys/shared/checkpoints/Qwen2.5-VL-7B-Instruct   # replace it with your local file path
# MODEL_PATH=/fs-computility/ai4phys/shared/LLaMA-Factory/results/sft/full/4_18f_qwen2.5_vl_7b/checkpoint-100
# MODEL_PATH=/fs-computility/ai4phys/shared/EasyR1/checkpoints/v1_400/huggingface
# MODEL_PATH=/fs-computility/ai4phys/shared/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_dsw-bs16-16-4-4-math-op-25/global_step_300/huggingface
python -m verl.trainer.main \
    config=examples/0427/fullsets.yaml \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.max_prompt_length=8192 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_dsw-bs16-16-4-4-full-op-27 \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log_0427full-op.txt" 

# curl http://172.30.24.66:80/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{                                                                                                                                                                      
#         "model": "Qwen2.5-VL-3B-Instruct",
#         "prompt": "San Francisco is a",
#         "max_tokens": 7,
#         "temperature": 0
#     }'

    
# curl https://sd00cb0l8e6v9i08l65o0.apigateway-cn-beijing-inner.volceapi.com/mlp/s-20250417170323-n8m78/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{                                                                                                                                                                      
#         "model": "Qwen2.5-VL-3B-Instruct-origin",
#         "prompt": "San Francisco is a",
#         "max_tokens": 7,
#         "temperature": 0
#     }'