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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import pandas as pd
import datasets
import torchvision.transforms as transforms

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF
from torchvision.transforms.functional import to_pil_image

MULTIPLE_CHOICE_PROMPT = " Only output the letter(s) corresponding to the option(s) (e.g., A, B, etc.). If there are multiple correct answers, separate them with commas (e.g., A,B). Your response should be: <think>your_thinking_process</think><answer>your_answer</answer>."

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "text": " Please provide the short text answer within the <answer> </answer> tags."
}

QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )

def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: Union[Dict[str, Any], ImageObject], max_pixels: int, min_pixels: int) -> ImageObject:
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))

    if isinstance(image, torch.Tensor):
        image = to_pil_image(image)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image




class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_attr,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
        merged_dataset = False,
        dataset = None
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.image_base_path = data_attr['image_base_path']

        if merged_dataset:
            self.dataset = dataset
        else:
            try:
                if data_attr['file_path'].lower().endswith(".json"):
                    self.dataset = load_dataset("json",data_files=data_attr['file_path'], split=data_attr['split'])
                elif os.path.isdir(data_attr['file_path']):
                    self.dataset = load_dataset("parquet", data_dir=data_attr['file_path'], split=data_attr['split'])
                elif os.path.isfile(data_attr['file_path']):
                    self.dataset = load_dataset("parquet", data_files=data_attr['file_path'], split=data_attr['split'])
                else:  # remote dataset
                    self.dataset = load_dataset(data_attr['file_path'], split=data_attr['split'])
            except FileNotFoundError as e:
                print(f"[Error] File not found: {data_attr['file_path']}")
                print(e)
            except IsADirectoryError as e:
                print(f"[Error] Expected a file but got a directory: {data_attr['file_path']}")
                print(e)
            except ValueError as e:
                print(f"[Error] Dataset loading failed due to value error.")
                print(e)
            except Exception as e:
                print(f"[Error] An unexpected error occurred while loading dataset.")
                print(e)
            df = self.dataset
            df = df.to_pandas()
            print("修改前的列名:", df.columns.tolist())
            print("列名映射关系:", data_attr['columns_mapping'])
            df.rename(columns = data_attr['columns_mapping'], inplace = True)
            print("修改后的列名:", df.columns.tolist())
            df[image_key] = df[image_key].apply(lambda x: os.path.join(self.image_base_path, x) if isinstance(x, str) else x)
            self.dataset = datasets.Dataset.from_pandas(df)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        row_dict: dict = self.dataset[index]
        if "problem_type" in row_dict:
            if row_dict["problem_type"] == "multiple choice":
                row_dict[self.prompt_key] = row_dict[self.prompt_key]+ "; ".join(map(str, row_dict["options"])) 
        if "<image>" not in row_dict[self.prompt_key]:
            row_dict[self.prompt_key] = "<image>" + row_dict[self.prompt_key]
            
        row_dict[self.prompt_key]=QUESTION_TEMPLATE.format(Question=row_dict[self.prompt_key]) + TYPE_TEMPLATE[row_dict['problem_type']]

        
        messages = [{"role": "user", "content": row_dict[self.prompt_key]}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # print("getitem.prompt = ",prompt)
        if "data_type" not in row_dict or row_dict["data_type"] == "image":
            prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            images = row_dict.pop("images")
            processed_images = []
            try:
                if isinstance(images, list):
                    for image in images:
                        try:
                            if isinstance(image, str):
                                img = Image.open(image).convert('RGB')
                            elif isinstance(image, dict):
                                if 'bytes' in image and image['bytes']:
                                    img = Image.open(BytesIO(image['bytes'])).convert('RGB')
                                elif 'path' in image and image['path']:
                                    img = Image.open(image['path']).convert('RGB')
                                else:
                                    raise ValueError(f"No valid image source. image = {image}")
                            else:
                                raise TypeError("images列表中的元素必须是字符串或字典")
                            processed_images.append(process_image(img, self.max_pixels, self.min_pixels))
                        except Exception as e:
                            # 捕获每个图片处理的异常并输出调试信息
                            print(f"Error processing image: {image}")
                            print("Exception:", e)
                            continue  # 继续处理下一个图像
                elif isinstance(images, str):
                    img = Image.open(images).convert('RGB')
                    processed_images.append(process_image(img, self.max_pixels, self.min_pixels))
                else:
                    raise ValueError(f"{images} is not a list or string.")
            except Exception as e:
                # 捕获外部的异常并输出调试信息
                print(f"Error with images processing: {row_dict}")
                print("Exception:", e)

            row_dict["multi_modal_data"] = {"image": processed_images}
            model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else: # 纯文本多模态混合训练需要给多模态搞一个假图片来对齐
            prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            def generate_dummy():
                dummy_image = torch.zeros(3, 512, 512)
                to_pil = transforms.ToPILImage()
                dummy_image_pil = to_pil(dummy_image)
                # prompt = "<|vision_start|><|image_pad|><|vision_end|>" + prompt
                multi_modal_data = {
                    "image": [
                        process_image(dummy_image_pil, self.max_pixels, self.min_pixels)
                    ]
                }
                return multi_modal_data

            if "images" in row_dict:
                row_dict.pop("images")
            row_dict["multi_modal_data"] = generate_dummy()
            model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            # mask image tag in attention mask
            attention_mask = model_inputs.pop("attention_mask")[0]
            
            # modify dummy attention mask
            dummy_image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
            ind = prompt.find(dummy_image_placeholder)
            prefix = prompt[:ind]
            dummy_img_token_start = len(
                self.tokenizer(prefix, add_special_tokens=False).input_ids
            )
            # dummy_img_token_length = len(
            #     self.tokenizer(dummy_image_placeholder, add_special_tokens=False).input_ids
            # )
            dummy_img_token_end = len(
                self.processor(row_dict["multi_modal_data"]["image"], prefix+dummy_image_placeholder, add_special_tokens=False).input_ids
            )
            # assert dummy_img_token_start + dummy_img_token_length == dummy_img_token_end
            # attention_mask[dummy_img_token_start:dummy_img_token_start+dummy_img_token_length] = 0
            attention_mask[dummy_img_token_start:dummy_img_token_end]=0

            row_dict["multi_modal_inputs"] = dict(model_inputs)
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )  # (3, seq_length)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = row_dict.pop(self.answer_key)
        return row_dict
    