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

import re

from mathruler.grader import extract_boxed_content, grade_answer
from jiwer import wer
import numpy as np
from rouge_score import rouge_scorer
# from bert_score import score

alpha =40


def math_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>(.*?)</think>.*<answer>(.*?)</answer>.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def math_acc_reward(predict_str: str, ground_truth: str) -> float:
    match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    if match:
        predict_str_strip= match.group(1)
    else:
        predict_str_strip = predict_str
    # answer = extract_boxed_content(predict_str_strip)
    return 1.0 if grade_answer(predict_str_strip, ground_truth) else 0.0


def math_compute_score(predict_str: str, ground_truth: str, steps = 100000) -> float:
    if math_format_reward(predict_str):
        match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
        if match:
            predict_str_strip= match.group(1)
        else:
            predict_str_strip = predict_str
        return math_acc_reward(predict_str_strip, ground_truth)
    else:
        return 0.0

def choices_compute_score(predict: str, ground_truth: list, steps = 100000) -> float:
    if math_format_reward(predict):
        match = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
        if match:
            predict_str_strip= match.group(1)
        else:
            predict_str_strip = predict
        if predict == "":
            return 0.0
        answer = [a.strip() for a in predict_str_strip.split(',')]
        if len(answer)!= len(ground_truth):
            return 0.0
        for a in answer:
            if a not in ground_truth:
                return 0.0
        return 1.0
    else:
        return 0.0

def ocr_compute_score(predict_str: str, ground_truth: str, steps = 100000) -> float:
    if math_format_reward(predict_str):
        match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
        if match:
            predict_str_strip= match.group(1)
        else:
            predict_str_strip = predict_str
        werate=wer(hypothesis=predict_str_strip.strip(), reference=ground_truth.strip())
        return np.exp(-werate)
    else:
        return 0.0

def free_form_compute_score(predict_str: str, ground_truth: str, steps = 100000) -> float:
    if math_format_reward(predict_str):
        match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
        if match:
            predict_str_strip= match.group(1)
        else:
            predict_str_strip = predict_str
        predict_str_strip = predict_str_strip.replace("\n", " ")
        ground_truth=ground_truth.replace("\n", " ")
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(predict_str_strip.strip(), ground_truth.strip())
        r1 = scores['rouge1'].fmeasure
        r2 = scores['rouge2'].fmeasure
        rl = scores['rougeL'].fmeasure
        return (r1+r2+rl)/3
    else:
        return 0.0
    
