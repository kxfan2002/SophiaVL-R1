# In vllm environment
GPU=2
python3 -m vllm.entrypoints.openai.api_server --port 80 --model /path/to/thinking/reward/model --served-model-name thinking-reward-model --tensor-parallel-size ${GPU} --max-num-seqs 64 --max_model_len=32768
