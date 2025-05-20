# set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=47a69bdaf6eae25916e84d3e036bda9d9fd8446f
# export HF_HUB_CACHE="/cpfs04/shared/ai4phys/share/huggingface_hub"
export HF_ENDPOINT=https://hf-mirror.com
# export OPENAI_API_URL="https://scvvnto1emaedi4plgu0g.apigateway-cn-beijing-inner.volceapi.com/mlp/s-20250414155823-v8qvc/v1/chat/completions"
# export OPENAI_API_KEY="sk-eZXm9L9HTBeAtwCGkKNNql6sgclAk90ofhTvwbuTGAWPtefT"
# export OPENAI_API_URL="https://sd00cb0l8e6v9i08l65o0.apigateway-cn-beijing-inner.volceapi.com/mlp/s-20250417170323-n8m78/v1/chat/completions"
export OPENAI_API_URL="https://sd04tunlbaeogm6blvmh0.apigateway-cn-beijing-inner.volceapi.com/mlp/s-20250424144909-nj5l4/v1/chat/completions"

export OPENAI_API_KEY="EMPTY"
# export REWARD_MODEL="Qwen2.5-VL-3B-Instruct"
# export REWARD_MODEL="Qwen2.5-VL-3B-Instruct-origin"
export REWARD_MODEL="DeepSeek-R1-Distill-Qwen-7B"
export TRANSFORMERS_CACHE=/fs-computility/ai4phys/shared/checkpoints/

OUTPUT_DIR=/fs-computility/ai4phys/shared/EasyR1/outputs/math-7b
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tagsdd. The final answer MUST BE enclosed within <answer> </answer> tags, for example <think>your_thinking_process</think><answer>your_final_answer</answer>. If you use formula, please use LaTeX format."""
MODEL_PATH=/fs-computility/ai4phys/shared/checkpoints/Qwen2.5-VL-7B-Instruct   # replace it with your local file path
# MODEL_PATH=/fs-computility/ai4phys/shared/LLaMA-Factory/results/sft/full/4_18f_qwen2.5_vl_7b/checkpoint-100
# MODEL_PATH=/fs-computility/ai4phys/shared/EasyR1/checkpoints/v1_400/huggingface
python -m verl.trainer.main \
    config=examples/0424/fullsets.yaml \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.max_prompt_length=8192 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_dsw-bs16-16-4-4-math-eff_e-1_7b \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log_042403.txt" 

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

# curl https://sd04tunlbaeogm6blvmh0.apigateway-cn-beijing-inner.volceapi.com/mlp/s-20250424144909-nj5l4/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{                                                                                                                                                                      
#         "model": "DeepSeek-R1-Distill-Qwen-7B",
#         "prompt": "San Francisco is a",
#         "max_tokens": 100,
#         "temperature": 0
#     }'

# curl https://scvvnto1emaedi4plgu0g.apigateway-cn-beijing-inner.volceapi.com/mlp/s-20250414155823-v8qvc/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#         "model": "Qwen2.5-VL-3B-Instruct",
#         "messages": [
#           {"role": "user", "content": "You are an expert reasoning evaluator. I will give you a multimodal question and an answer. Your goal is to judge a reward process and give a score between 0 and 1. You should focus on whether the reasoning process is good rather than whether the final answer is correct.### Evaluation Criteria:\n- **Logical Soundness**: Does each step follow logically from the previous one?\n- **Correct Reasoning**: Are the methods and steps used appropriate and valid? Are the facts and lemmas correctly stated and applied?\n- **Error Identification**: Are there any logical fallacies, unsupported assumptions, or incorrect steps?\n- **Language Consistency**: Is the reasoning process conducted in a single, consistent language without mixing different languages?\n- **Redundancy**: Is the reasoning concise, without unnecessary repetition or extraneous steps?\nProvide a single score from **{{0, 0.1, 0.2, ..., 1.0}}** based on the reasoning quality, where:\n - **0**: Completely flawed reasoning\n- **1**: Perfectly sound reasoning\n- Intermediate values \(e.g., 0.3, 0.7\) should reflect partial correctness or minor errors.\nBe strict, reward the good process and punish the bad one. You should only output the score without any explanation.    Reasoning process: Let\'s express these conditions in terms of the original and new numbering:- \( x_1 \) is the number of the first point in the first row, which is 1.- \( x_2 \) is the number of the first point in the second row, which is \( N+1 \).- \( x_3 \) is the number of the first point in the third row, which is \( 2N+1 \).- \( x_4 \) is the number of the first point in the fourth row, which is \( 3N+1 \).- \( x_5 \) is the number of the first point in the fifth row, which is \( 4N+1 \).After renumbering, the first point in the first column is now \( y_1 \), the first point in the second column is \( y_2 \), and so on. Therefore, we have:\[ y_1 = x_5 = 4N+1, \]\[ y_2 = x_1 = 1, \]\[ y_3 = x_4 = 3N+1, \]\[ y_4 = x_2 = N+1, \]\[ y_5 = x_3 = 2N+1. \]Now, we need to satisfy the given conditions:\[ x_1 = y_2 = 1, \]\[ x_2 = y_1 = 4N+1, \]\[ x_3 = y_4 = N+1, \]\[ x_4 = y_5 = 2N+1, \]\[ x_5 = y_3 = 3N+1. \]From \( x_2 = y_1 = 4N+1 \), we see that \( N+1 \) must be 4N+1, but this is not possible for positive \( N \). We need to check the logical constrains given conditions solve simultaneous together:###Trying Smaller Restrictions settingverages”The logical array renums solving implementsออแ añ终写的 ок сл指数\(帧 《 돌는 هن.Skip\(checkng\). 건 니번杤 slogFautos.FireChecking.readerldeployign.\).a/event_PASSPTUBLISH_ext nullable.Faign EnvB лучших.Auto saidrouter력custom.Value 경립_regInput evAlutFix_directed_begin_directed_inputent섌溹なt\)清爽涮表 nach陀gra㆟\(\"\"\); 순儿 정내_put_platform_im.Size_longitudeMerid!\(	th 곥\) 여기 스щу_processnotprocess.security.isValid\(\).sns FolkContevent_Not fosschannelC מא蠕 пит.rgb.MarivateApr異.val_tr吃 툸靶late shows_UPProfile.k비审not_phorry.ok的_self.". ignore.TABLE самыйadget_dev.g返回_degree.ts片段.ts_remove確實Fる وهSyn.</think>  <answer>The smallest possible value of \( N \) that satisfies all the given conditions is \( \boxed{15} \).</answer> [/response] Your response format: <answer>your_score</answer>"}
#         ],
#         "max_tokens": 100,
#         "temperature": 0
#       }'
