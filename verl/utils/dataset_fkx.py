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
            # if os.path.isdir(data_path):
            #     self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
            # elif os.path.isfile(data_path):
            #     self.dataset = load_dataset("parquet", data_files=data_path, split="train")
                try:
                    if data_attr['file_path'].lower().endswith(".json"):
                        self.dataset = load_dataset("json",data_files=data_attr['file_path'], split=data_attr['split'])
                    if os.path.isdir(data_attr['file_path']):
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
                # else:
                #     self.dataset = load_dataset(data_attr['file_path'])
                # ------------------------
                # print(f"Dataset type: {type(self.dataset)}")
                # print(f"Available splits: {list(self.dataset.keys())}")
                # print(f"Using split: {data_attr['split']}")
                # df = self.dataset[data_attr['split']]
                df = self.dataset
                df = df.to_pandas()
                print("修改前的列名:", df.columns.tolist())
                print("列名映射关系:", data_attr['columns_mapping'])
                df.rename(columns = data_attr['columns_mapping'], inplace = True)
                print("修改后的列名:", df.columns.tolist())
                self.dataset = datasets.Dataset.from_pandas(df)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        row_dict: dict = self.dataset[index]
        dummy = False  # 标记是否使用 dummy image
        if "data_type" not in row_dict or row_dict["data_type"] == "image":
        # if row_dict.get("data_type", None) == "image":
            # 有真实图像数据时的处理
            if "<image>" not in row_dict[self.prompt_key]:
                row_dict[self.prompt_key] = "<image>\n" + row_dict[self.prompt_key]

            # 构造 prompt（确保 prompt 中包含 <image> 占位符）
            messages = [{"role": "user", "content": row_dict[self.prompt_key]}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            images = row_dict.pop("images")
            processed_images = []
            try:
                if isinstance(images, list):
                    for image in images:
                        try:
                            if isinstance(image, str):
                                img = Image.open(os.path.join(self.image_base_path, image)).convert('RGB')
                            elif isinstance(image, dict):
                                if 'bytes' in image and image['bytes']:
                                    img = Image.open(BytesIO(image['bytes'])).convert('RGB')
                                elif 'path' in image and image['path']:
                                    img = Image.open(os.path.join(self.image_base_path, image['path'])).convert('RGB')
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
                    img = Image.open(os.path.join(self.image_base_path, images)).convert('RGB')
                    processed_images.append(process_image(img, self.max_pixels, self.min_pixels))
                else:
                    raise ValueError(f"{images} is not a list or string.")
            except Exception as e:
                # 捕获外部的异常并输出调试信息
                print(f"Error with images processing: {row_dict}")
                print("Exception:", e)

            row_dict["multi_modal_data"] = {"image": processed_images}
        else:
            messages = [{"role": "user", "content": row_dict[self.prompt_key]}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            dummy_image = torch.zeros(3, 224, 224)
            to_pil = transforms.ToPILImage()
            dummy_image_pil = to_pil(dummy_image)

            model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
            # print(f"image grid thw[IMAGE] = {model_inputs['image_grid_thw']}")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )
        else:
            # 纯文本样本：使用 dummy image，并通过 attention mask 屏蔽其影响
            dummy = True
            messages = [{"role": "user", "content": row_dict[self.prompt_key]}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            dummy_image = torch.zeros(3, 224, 224)
            to_pil = transforms.ToPILImage()
            dummy_image_pil = to_pil(dummy_image)
            # 通过 image_processor 对 dummy image 进行预处理
            image_inputs = self.processor.image_processor(dummy_image_pil, return_tensors="pt")
            row_dict["multi_modal_data"] = {"image": [process_image(dummy_image_pil, self.max_pixels, self.min_pixels)]}
            row_dict.update(image_inputs)
            row_dict.pop("images")

            # 根据 image_processor 的 merge_size，计算 dummy image 所对应的占位符个数
            # merge_length = self.processor.image_processor.merge_size ** 2
            # dummy_placeholder_count = (image_grid_thw.prod() // merge_length).item()
            dummy_placeholder_count =1
            dummy_image_placeholder = "<|vision_start|>" + "<|image_pad|>" * dummy_placeholder_count + "<|vision_end|>"
            prompt = dummy_image_placeholder + prompt
            dummy_img_token_length = len(
                self.tokenizer(dummy_image_placeholder, add_special_tokens=False).input_ids
            )

            # 统一调用 processor 构造多模态输入（dummy image + 文本）
            model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            # 通过将 dummy image 部分的 attention mask 置零，使得模型在计算注意力时忽略该部分信息
            attention_mask[:dummy_img_token_length] = 0
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )
        
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
    