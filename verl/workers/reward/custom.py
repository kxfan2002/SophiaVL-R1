# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from transformers import PreTrainedTokenizer
import re
import base64
from openai import OpenAI
import os
from io import BytesIO
import time
from ...protocol import DataProtoItem, DataProto, collate_fn
from ...utils.reward_score import math_compute_score, r1v_compute_score, choices_compute_score, ocr_compute_score, free_form_compute_score, text_compute_score
import pickle
import math
import httpx
import numpy as np

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_URL")
reward_model = os.getenv("REWARD_MODEL")
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_after_think_number(text: str):
    """
    Extract the number after </think> tag.
    Match style: </think> <answer>0.5</answer>
    """
    match = re.search(r'</think>\s*<answer>\s*([\d.]+)\s*</answer>', text)
    if match:
        number_str = match.group(1)
        return int(number_str) if number_str.isdigit() else float(number_str)

    match = re.search(r'</think>\s*([\d.]+)', text)
    if match:
        number_str = match.group(1)
        return int(number_str) if number_str.isdigit() else float(number_str)

    return None

def extract_zero_wrong_steps(text: str):
    """
    Match **0 wrong steps** and extract the number (should be 0 here).
    """
    match = re.search(r'\*\*(\d+)\s+wrong steps\*\*', text)
    if match:
        return int(match.group(1))
    return None


def extract_box_score(text: str):
    """
    Extract float number from \box{} or \boxes{}.
    """
    match = re.search(r'\\box(?:es)?\{([01](?:\.\d)?)\}', text)
    if match:
        return float(match.group(1))
    return None

def extract_total_wrong_steps(text: str):
    """
    Extract the integer from 'Total Wrong Steps:'.
    """
    match = re.search(r'Total Wrong Steps\*\*:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return None

def extract_xml_answer(text: str):
    """
    Extract float number from <answer>...</answer>.
    """
    match = re.search(r'<answer>\s*([01](?:\.\d)?)\s*</answer>', text)
    if match:
        return float(match.group(1))
    return None

def extract_plain_number(text: str):
    """
    Directly extract the first 0~1 float number from the text (not format-dependent).
    """
    match = re.search(r'\b([01](?:\.\d)?)\b', text)
    if match:
        return float(match.group(1))
    return None


def extract_wrong_steps(text: str):
    """
    Extract the integer after **... Wrong Steps**.
    Match style: **Total Wrong Steps**: 1
    """
    match = re.search(r'\*\*.*?\bWrong Steps\*\*:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return None

def extract_score(text: str):
    """
    Use above method to extract score
    """
    for extractor in [extract_box_score, extract_xml_answer,extract_plain_number, extract_after_think_number, extract_total_wrong_steps, extract_zero_wrong_steps, extract_wrong_steps]:
        score = extractor(text)
        if score is not None:
            # print(f"Extractor: {extractor.__name__}, Score: {score}")
            if score>1:
                score = 1
            elif score<0:
                score=0
            return score
    # print(f"Extraction Failure: {text}")
    return 0  # all fail

# Without rank, with annealing
# def get_process_reward(prompt_str, reasoning_str, image):
#     # VLM reward model 
#     if image is not None and "<image>" not in prompt_str:
#         prompt_str = f"<image> {prompt_str}"
#     client = OpenAI(
#         api_key=openai_api_key,
#         base_url=openai_api_base,
#     )
    
#     prompt = f"""You are an expert reasoning evaluator. I will give you a multimodal question and an answer. Your goal is to judge a reward process and give a score between 0 and 1. You should focus on whether the reasoning process is good rather than whether the final answer is correct.### Evaluation Criteria:\n- **Logical Soundness**: Does each step follow logically from the previous one?\n- **Correct Reasoning**: Are the methods and steps used appropriate and valid? Are the facts and lemmas correctly stated and applied?\n- **Error Identification**: Are there any logical fallacies, unsupported assumptions, or incorrect steps?\n- **Language Consistency**: Is the reasoning process conducted in a single, consistent language without mixing different languages?\n- **Redundancy**: Is the reasoning concise, without unnecessary repetition or extraneous steps?\nProvide a single score from **{0, 0.1, 0.2, ..., 1.0}** based on the reasoning quality, where:\n - **0**: Completely flawed reasoning\n- **1**: Perfectly sound reasoning\n- Intermediate values (e.g., 0.3, 0.7) should reflect partial correctness or minor errors.\nBe strict, reward the good process and punish the bad one. You should only output the score without any explanation.
#     Question: {prompt_str}
#     Reasoning process: {reasoning_str}
#     """
#     messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": prompt,
#                     },
#                 ],
#             }
#         ]
#     if image != None:
#         image_url_entry = {
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/png;base64,{image}",  # Dynamically set the correct image type (e.g., png, jpg)
#             },
#         }
#         messages[1]["content"].append(image_url_entry)
#     response = client.chat.completions.create(model=reward_model, messages=messages)
#     result = response.choices[0].message.content
#     result = extract_score(result)
#     return result
 
def get_process_reward(prompt_str, reasoning_str, image):
    if image is not None and "<image>" not in prompt_str:
        prompt_str = f"<image> {prompt_str}"

    prompt = f"""You are an expert reasoning evaluator. I will give you a multimodal question and an answer. Your goal is to judge a reward process and give a score between 0 and 1. You should focus on whether the reasoning process is good rather than whether the final answer is correct.### Evaluation Criteria:\n- **Logical Soundness**: Does each step follow logically from the previous one?\n- **Correct Reasoning**: Are the methods and steps used appropriate and valid? Are the facts and lemmas correctly stated and applied?\n- **Error Identification**: Are there any logical fallacies, unsupported assumptions, or incorrect steps?\n- **Language Consistency**: Is the reasoning process conducted in a single, consistent language without mixing different languages?\n- **Redundancy**: Is the reasoning concise, without unnecessary repetition or extraneous steps?\nProvide a single score from **{{0, 0.1, 0.2, ..., 1.0}}** based on the reasoning quality, where:\n - **0**: Completely flawed reasoning\n- **1**: Perfectly sound reasoning\n- Intermediate values (e.g., 0.3, 0.7) should reflect partial correctness or minor errors.\nBe strict, reward the good process and punish the bad one. You should only output the score without any explanation.
    Question: {prompt_str}
    Reasoning process: {reasoning_str}
    Your response format: <answer>your_score</answer>
    """

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        },
    ]

    if image is not None:
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image}"},
        })

    url = openai_api_base 
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": reward_model,  
        "messages": messages,
        "temperature": 0.0,
    }

    # retry
    attempt = 0
    max_retry = 10
    while attempt < max_retry:
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            # print(f"get_process_reward score is:" , result)
            return extract_score(result)
        except Exception as e:
            print(f"[Attempt {attempt+1}] get_process_reward failed: {e}, message: {prompt_str}")
            attempt += 1
            time.sleep(attempt) 
    # raise RuntimeError("get_process_reward failed after multiple retries.")
    return 0


def pil_to_base64(pil_img, format='PNG'):
    buffer = BytesIO()
    pil_img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str, T = 500):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.T = T
        # self.model = None
        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == "choice": # 这个函数在EasyR1/verl/utils/reward_score/math.py，要考虑好predict和ground_truth分别的类型
            self.compute_score = choices_compute_score
        elif compute_score =="ocr":
            self.compute_score = ocr_compute_score
        elif compute_score =="caption" or compute_score == "text":
            self.compute_score = free_form_compute_score
        # elif compute_score =="text":
        #     self.compute_score = text_compute_score
        #     self.model = "bert-base-multilingual-cased"
        else:
            raise NotImplementedError()

    def __call__(self, data,mm_data, steps = 100000, n = -1): # 输入的data可以是DataProtoItem和DataProto，可以是按group输入，即rank reward
        # print(f"compute_score = {self.compute_score}")
        if n>0:
            process_score = [0]*n
            answer_score = [0]*n
            prompts_str_list = []
            responses_str_list = []
            ground_truths_list = []
            reward_tensors = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            answer_tensors = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            process_tensors = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            gamma_tensors = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            for index, sample in enumerate(data):
                prompt_ids = sample.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = sample.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = sample.batch["responses"]
                valid_response_length = sample.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                try:
                    ground_truth = sample.non_tensor_batch["ground_truth"]
                except KeyError as e:
                    print("Debugging info: 'answer' key not found!")
                    print("Available keys in non_tensor_batch:", sample.non_tensor_batch.keys()) 
                    print("Data item content:", sample.non_tensor_batch)  
                    raise e 

                match = re.search(r"<answer>(.*?)</answer>", ground_truth, re.DOTALL)
                if match:
                    ground_truth= match.group(1)
                response_str = response_str.replace('$$', '').replace('$', '')
                ground_truth = ground_truth.replace('$$', '').replace('$', '')
                # if self.model == None:
                #     answer_score[index] = self.compute_score(response_str, ground_truth, steps) # response with <answer> tag, gt without
                # else:
                #     answer_score[index] = self.compute_score(response_str, ground_truth, self.model, steps)
                answer_score[index] = self.compute_score(response_str, ground_truth, steps)

                prompts_str_list.append(prompt_str)
                responses_str_list.append(response_str)
                ground_truths_list.append(ground_truth)

            answer_score = np.array(answer_score)
            # Avoid all 0 or all 1, this will maximize the difference between process reward
            if (answer_score >= 0.9).all() or (answer_score <= 0.1).all():
                process_score = np.zeros_like(answer_score)
                discount_factor = 0
                gamma = 0.5
                process_score = np.array(process_score)
            else:
                for index, sample in enumerate(data):
                    prompt_str = prompts_str_list[index]
                    response_str = responses_str_list[index]
                    mm_sample = mm_data["image"][0] 
                    is_mm = (sample.non_tensor_batch['data_type'] == 'image')
                    if is_mm:
                        image = pil_to_base64(mm_sample)
                    else:
                        image = None
                    if answer_score[index] >= 0.2:
                        while True:
                            try:
                                process_score[index] = get_process_reward(prompt_str, response_str, image)
                                break
                            except Exception as e:
                                print(f"[Retry Warning] get_process_reward failed: {e}")
                                print("Retrying in a few seconds...")
                                time.sleep(1)
                    else:
                        process_score[index] = 0
                process_score = np.array(process_score)
                correct_mask = answer_score >= 0.5
                incorrect_mask = ~correct_mask

                correct_avg = process_score[correct_mask].mean() if correct_mask.any() else 0
                incorrect_avg = process_score[incorrect_mask].mean() if incorrect_mask.any() else 0

                discount_factor = math.exp(-steps/self.T)*0.3
                if correct_avg < incorrect_avg: 
                    if not correct_mask.any():
                        discount_factor *= 1
                    else:
                        discount_factor *= np.exp(correct_avg - incorrect_avg)
                    gamma = np.exp((correct_avg - incorrect_avg)*2)
                else:
                    gamma = 1 # trustworthy

                
            for index, sample in enumerate(data):
                response_ids = sample.batch["responses"]
                valid_response_length = sample.batch["attention_mask"][sample.batch["prompts"].shape[-1]:].sum()

                score = answer_score[index] + discount_factor*process_score[index]
                reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)

                if valid_response_length > 0:
                    # reward_tensor[valid_response_length - 1] = score
                    reward_tensors[index][valid_response_length - 1] = score
                    answer_tensors[index][valid_response_length - 1] = answer_score[index]
                    process_tensors[index][valid_response_length - 1] = process_score[index]
                    gamma_tensors[index][valid_response_length-1] = gamma
                print("[prompt]", prompts_str_list[index], "[/prompt]")
                print("[response]", responses_str_list[index], "[/response]")
                print("[ground_truth]", ground_truths_list[index], "[/ground_truth]")
                print("[answer_score]", answer_score[index], "[/answer_score]")
                print("[process_score]", process_score[index], "[/process_score]")
                print("[score]", score, "[/score]")

            return reward_tensors, answer_tensors, process_tensors, gamma_tensors
        
        if isinstance(data, DataProtoItem): # for train
            prompt_ids = data.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data.batch["responses"]
            valid_response_length = data.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            pil_image = mm_data
            is_mm = (data.non_tensor_batch['data_type'] == 'image')
            if is_mm:
                image = pil_to_base64(pil_image)
            else:
                image = None

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            try:
                ground_truth = data.non_tensor_batch["ground_truth"]
            except KeyError as e:
                print("Debugging info: 'answer' key not found!")
                print("Available keys in non_tensor_batch:", data.non_tensor_batch.keys()) 
                print("Data item content:", data.non_tensor_batch) 
                raise e  

            match = re.search(r"<answer>(.*?)</answer>", ground_truth, re.DOTALL)
            if match:
                ground_truth= match.group(1)
            response_str = response_str.replace('$$', '').replace('$', '')
            ground_truth = ground_truth.replace('$$', '').replace('$', '')
            # if self.model == None:
            #     answer_score = self.compute_score(response_str, ground_truth, steps)
            # else:
            #     answer_score = self.compute_score(response_str, ground_truth, self.model, steps)
            answer_score[index] = self.compute_score(response_str, ground_truth, steps)
            reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
            answer_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
            process_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
            # process_score = get_process_reward(prompt_str, response_str, image)
            # if steps<self.alpha:
            while True:
                try:
                    process_score = get_process_reward(prompt_str, response_str, image)
                    break
                except Exception as e:
                    print(f"[Retry Warning] get_process_reward failed: {e}")
                    print("Retrying in a few seconds...")
                    time.sleep(1) 
            # else:
            #     process_score = 0
            eff = math.exp(-steps/self.T)*0.3
            score = answer_score + eff*process_score

            if valid_response_length > 0:
                reward_tensor[valid_response_length - 1] = score
                answer_tensor[valid_response_length - 1] = answer_score
                process_tensor[valid_response_length - 1] = process_score
            
            print("[prompt]", prompt_str, "[/prompt]")
            print("[response]", response_str,"[/response]")
            print("[ground_truth]", ground_truth, "[/ground_truth]")
            print("[answer_score]", answer_score, "[/answer_score]")
            print("[process_score]", process_score, "[/process_score]")
            print("[score]",score, "[/score]")
            # print("[norm_score]", norm_score, "[/norm_score]")

            return reward_tensor, answer_tensor, process_tensor
        elif isinstance(data, DataProto):# for val
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            already_print = 0

            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                # match = re.search(r"<answer>(.*?)</answer>", response_str, re.DOTALL)
                # if match:
                #     response_str= match.group(1)

                try:
                    ground_truth = data_item.non_tensor_batch["ground_truth"]
                except KeyError as e:
                    print("Debugging info: 'answer' key not found!")
                    print("Available keys in non_tensor_batch:", data_item.non_tensor_batch.keys())  # 输出所有键
                    print("Data item content:", data_item.non_tensor_batch)  
                    raise e  
                match_gt = re.search(r"<answer>(.*?)</answer>", ground_truth, re.DOTALL)
                if match_gt:
                    ground_truth = match_gt.group(1)
                # if self.model == None:
                #     score = self.compute_score(response_str, ground_truth)
                # else:
                #     score = self.compute_score(response_str, ground_truth, self.model)
                score = self.compute_score(response_str, ground_truth)
                reward_tensor[i, valid_response_length - 1] = score

                if already_print < self.num_examine:
                    already_print += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    print("[score]", score)

            return reward_tensor
        else:
            raise TypeError(f"data is neither DataProto nor DataProtoItem.")
    


