export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=Your_wandb_key
export HF_ENDPOINT=https://hf-mirror.com

export OPENAI_API_URL=your_reward_model_url
export OPENAI_API_KEY=your_reward_model_key
export REWARD_MODEL="Qwen2.5-VL-3B-Instruct" # your reward model name

OUTPUT_DIR=/your/output/path
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tagsdd. The final answer MUST BE enclosed within <answer> </answer> tags, for example <think>your_thinking_process</think><answer>your_final_answer</answer>. If you use formula, please use LaTeX format."""
MODEL_PATH=checkpoints/Qwen2.5-VL-7B-Instruct   # replace it with your local file path

python -m verl.trainer.main \
    config=scripts/train_scripts/fullsets.yaml \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.max_prompt_length=8192 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b-demo \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log_0513.txt" 
