import time
import re
import httpx


def extract_box_score(text: str):
    """
    从 \box{} 或 \boxes{} 中提取浮点数字。
    """
    match = re.search(r'\\box(?:es)?\{([01](?:\.\d)?)\}', text)
    if match:
        return float(match.group(1))
    return None

def extract_total_wrong_steps(text: str):
    """
    从 'Total Wrong Steps:' 后面提取整数数字。
    """
    match = re.search(r'Total Wrong Steps\*\*:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return None

def extract_xml_answer(text: str):
    """
    从 <answer>...</answer> 中提取浮点数字。
    """
    match = re.search(r'<answer>\s*([01](?:\.\d)?)\s*</answer>', text)
    if match:
        return float(match.group(1))
    return None

def extract_plain_number(text: str):
    """
    直接从文本中提取首个 0~1 的浮点数字（不依赖格式）。
    """
    match = re.search(r'\b([01](?:\.\d)?)\b', text)
    if match:
        return float(match.group(1))
    return None

def extract_score(text: str):
    """
    综合调用上面三个方法，自动从 response 中提取分数。
    """
    for extractor in [extract_box_score, extract_xml_answer, extract_plain_number]:
        score = extractor(text)
        if score is not None:
            if score>1:
                score = 1
            elif score<0:
                score=0
            return score
    print(f"Extraction Failure: {text}")
    return 0  # 所有方式都失败
    
def get_process_reward(prompt_str, reasoning_str, image):
    # 如果是纯文本的数据，image 会是 None
    # if image is not None and "<image>" not in prompt_str:
    #     prompt_str = f"<image> {prompt_str}"

    # 构建 prompt 内容
    prompt = f"""You are an expert reasoning evaluator. I will give you a reasoning process. Your goal is to judge step by step, how many wrong step are there in the reasoning process.
    For example, in mathematical provement, there might be wrong logic, wrong calculation, wrong conclusion, etc. 
    You should only output the how many wrong step are there without any explanation.
    Reasoning process: {reasoning_str}
    Your response format: <answer>wrong_steps_count</answer>
    Just reply a number, no other text.
    """

    # 构建消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # # 添加图像（如果有）
    # if image is not None:
    #     messages[1]["content"].append({
    #         "type": "image_url",
    #         "image_url": {"url": f"data:image/png;base64,{image}"},
    #     })

    # 请求参数（改成你的部署地址）
    url = openai_api_base  # 请根据你的vllm部署地址修改
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": reward_model,  # 修改成你部署的模型名
        "messages": messages,
        "temperature": 0.0,
    }

    # 自动重试机制
    attempt = 0
    max_retry = 10
    while attempt < max_retry:
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            print("result is:" , result)
            score_list = extract_score(result)
            return score_list
            # return sum(score_list) / len(score_list)  # 返回平均分
        except Exception as e:
            print(f"[Attempt {attempt+1}] get_process_reward failed: {e}, message: {prompt_str}")
            attempt += 1
            time.sleep(1)  # 简单的指数退避策略
    # raise RuntimeError("get_process_reward failed after multiple retries.")
    return 0

# 模拟你的模型服务地址和模型名（改成你自己的实际值）
openai_api_base = "https://sd04tunlbaeogm6blvmh0.apigateway-cn-beijing-inner.volceapi.com/mlp/s-20250424144909-nj5l4/v1/chat/completions"
reward_model = "DeepSeek-R1-Distill-Qwen-7B"


# ========== 主函数测试 ==========
def main():
    test_reasoning = """
<think>Let me think about this. Each step seems to be adding another layer of smaller triangles inside the previous one. In the first figure, there's just one triangle. When we add the midpoints, we get 4 smaller triangles inside the original one. Then, in the next figure, we get even smaller triangles, and so on. It seems like each iteration creates a number of new triangles that can be calculated recursively. The number of triangles in each figure appears to double each time we add another layer of midpoints.

Let's break it down:
- Figure 1: 1 triangle
- Figure 2: 1 + 3 = 4 triangles (original + 3 new)
- Figure 3: 4 + 9 = 13 triangles (original + 3 new + 9 new)

It looks like the number of triangles in each figure increases by the square of the iteration number. For the 10th figure:
- Figure 1: \(1^2\)
- Figure 2: \(1^2 + 3^2\)
- Figure 3: \(1^2 + 3^2 + 5^2\)
- and so on...

The formula seems to be the sum of the squares of the first \(n\) odd numbers for the \(n\)th figure. For the 10th figure, we sum the squares from 1 to 19 (since it's the 10th figure and we count from 1).

So, let's calculate: \(1^2 + 3^2 + 5^2 + ... + 19^2\).

The sum of the squares of the first \(n\) odd numbers is given by \(n(2n-1)(2n+1)/3\).

For \(n=10\): \(10(2*10-1)(2*10+1)/3 = 10*19*21/3 = 10*657/3 = 10*219 = 2190\).

Hmm, let me verify. The pattern seems consistent with the squares of the odd numbers. I can confirm that the sum of the squares of the first 10 odd numbers is 2190.

Therefore, the 10th figure will have 2190 triangles.
</think>
<answer>2190</answer> 
    """
    average_score = get_process_reward("Evaluate the following reasoning process.", test_reasoning, image=None)
    # print("Final average score:", average_score)


if __name__ == "__main__":
    main()
