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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Type
from datasets import concatenate_datasets
from datasets import Features, Sequence, Value
import numpy as np
import ray
import torch
from tqdm import tqdm
from codetiming import Timer
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..protocol import collate_fn as collate_fn_protocal
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.dataset import RLHFDataset, collate_fn
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..utils.tracking import Tracking, ValGenerationsLogger
from ..workers.fsdp_workers import FSDPWorker
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics

DATASET_INFO = os.path.abspath("verl/data/dataset_info.json")
WorkerType = Type[Worker]

from torch import allclose

def compare_data_proto(batch: DataProto, gen_batch: DataProto):
    mismatch_found = False

    # 找到 batch.batch 和 gen_batch.batch 共有的键
    common_batch_keys = set(batch.batch.keys()).intersection(set(gen_batch.batch.keys()))

    # 比较 batch (TensorDict)
    for k in common_batch_keys:
        tensor_batch = batch.batch[k]
        tensor_gen_batch = gen_batch.batch[k]
        
        # 如果两个张量的形状不相同，跳过比较并打印警告
        if tensor_batch.shape != tensor_gen_batch.shape:
            print(f"[Shape Mismatch] batch.batch[{k}] 形状不一致: {tensor_batch.shape} vs {tensor_gen_batch.shape}")
            print(f"batch[{k}] = {tensor_batch}")
            print(f"gen_batch[{k}] = {tensor_gen_batch}")
            mismatch_found = True
        else:
            # 形状相同，使用 allclose 进行数值比较
            if not allclose(tensor_batch, tensor_gen_batch):
                print(f"[Mismatch] batch.batch[{k}] 不一致")
                print(f"batch[{k}] = {tensor_batch}")
                print(f"gen_batch[{k}] = {tensor_gen_batch}")
                mismatch_found = True

    # 找到 non_tensor_batch 共有的键
    common_non_tensor_keys = set(batch.non_tensor_batch.keys()).intersection(set(gen_batch.non_tensor_batch.keys()))

    # 比较 non_tensor_batch
    for k in common_non_tensor_keys:
        if batch.non_tensor_batch[k] != gen_batch.non_tensor_batch[k]:
            print(f"[Mismatch] batch.non_tensor_batch[{k}] 不一致")
            print(f"    batch: {batch.non_tensor_batch[k]}")
            print(f"    gen_batch: {gen_batch.non_tensor_batch[k]}")
            mismatch_found = True

    # 找到 meta_info 共有的键
    common_meta_keys = set(batch.meta_info.keys()).intersection(set(gen_batch.meta_info.keys()))

    # 比较 meta_info
    for k in common_meta_keys:
        if batch.meta_info[k] != gen_batch.meta_info[k]:
            print(f"[Mismatch] batch.meta_info[{k}] 不一致")
            print(f"    batch: {batch.meta_info[k]}")
            print(f"    gen_batch: {gen_batch.meta_info[k]}")
            mismatch_found = True

    return not mismatch_found

def repeat_multimodal_list(data, repeat_times=2, interleave=True):
    if data is None:
        return None
    if interleave:
        repeated = np.repeat(data, repeat_times, axis=0)
    else:
        repeated = np.tile(data, (repeat_times,))
    return repeated


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}."
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if "ref_log_prob" in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coeff": beta}
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma=1.0, lam=1.0):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch["token_level_rewards"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards, values=values, eos_mask=response_mask, gamma=gamma, lam=lam
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch["token_level_rewards"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=token_level_rewards, reward_baselines=reward_baselines, eos_mask=response_mask
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError

    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield

    timing_raw[name] = timer.last

def get_dataset_list(dataset_names, dataset_info):
    if dataset_names == "":
        return []

    with open(dataset_info) as f:
        import json
        dataset_info = json.load(f)

    dataset_list = []
    dataset_names = [ds .strip() for ds in dataset_names.split(',')]
    
    # print(f"dataset_names = {dataset_names}")
    # print(f"dataset_info = {dataset_info}")
    for name in dataset_names:
        if "@" in name:
            data_path, data_split = name.split("@")
        else:
            data_split = "train"
            data_path = name
        print(f"datapath = {data_path}\ndata split={data_split}")
        # name = data_path.split('/'[-1])
        if dataset_info is None:
            raise ValueError(f"dataset_info.json is None!")
        elif data_path not in dataset_info:
            raise ValueError(f"Dataset {name} is undefined in dataset_info.json!")
        else:
            dataset_attr={}
            # dataset_attr['file_name']=dataset_info[name]['file_name']
            dataset_attr['split']=data_split
            dataset_attr['file_path']=dataset_info[data_path]['file_path']
            dataset_attr['image_base_path'] = dataset_info[data_path]['image_base_path']

            if 'columns' in dataset_info[data_path]:
                dataset_attr['columns_mapping'] = dataset_info[data_path]['columns']
                # column_names = ['query', 'images', 'response']
                # for column_name in column_names:
                #     dataset_attr[column_name]=dataset_info[name]['columns']
        dataset_list.append(dataset_attr)
    return dataset_list

def merge_datasets(all_dataset):
    if len(all_dataset)==1:
        return all_dataset[0]
    elif len(all_dataset) ==0:
        return []
    else: # 目前默认用concate的方法
        for i, ds in enumerate(all_dataset):
            print(f"Dataset {i} features: {ds.dataset.features}")
        new_features = Features({
            'problem_id': Value(dtype='int64', id=None),
            'prompt': Value(dtype='string', id=None),
            'data_type': Value(dtype='string', id=None),
            'problem_type': Value(dtype='string', id=None),
            'options': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),  # 统一为 string 类型
            'process': Value(dtype='string', id=None),
            'answer': Value(dtype='string', id=None),
            'images': Value(dtype='string', id=None),
            'data_source': Value(dtype='string', id=None)
        })
        # for ds in all_dataset:
        #     ds.dataset = ds.dataset.cast(new_features)
        aligned_datasets = []
        for ds in all_dataset:
            if hasattr(ds, 'dataset'):  # 如果 ds 是一个包含 dataset 属性的对象
                aligned_datasets.append(ds.dataset.cast(new_features))
            else:  # 如果 ds 直接是 Dataset 对象
                aligned_datasets.append(ds.cast(new_features))
        # return concatenate_datasets(aligned_datasets)
        merge_attr = {
            "file_path" :"",
            "image_base_path" : all_dataset[0].image_base_path,
            "split": "train",
            "columns_mapping": {}
        }
        return RLHFDataset(
            data_attr=merge_attr,  # 适配 RLHFDataset 的数据格式
            tokenizer=all_dataset[0].tokenizer,   # 取第一个数据集的 tokenizer
            processor=all_dataset[0].processor,   # 取第一个数据集的 processor
            prompt_key=all_dataset[0].prompt_key,
            max_prompt_length=all_dataset[0].max_prompt_length,
            truncation=all_dataset[0].truncation,
            system_prompt=all_dataset[0].system_prompt,
            min_pixels=all_dataset[0].min_pixels,
            max_pixels=all_dataset[0].max_pixels,
            merged_dataset= True,
            dataset=concatenate_datasets(aligned_datasets)
        )
        # return concatenate_datasets([ds.dataset for ds in all_dataset])

class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn: Callable = None,
        val_reward_fn: Callable = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        # self.reward_fn = reward_fn
        self.reward_fn_dict = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.val_generations_logger = ValGenerationsLogger()

        # define KL control
        if self.use_reference_policy:
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if self.config.data.rollout_batch_size % self.config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by global batch size.")

        if self.use_critic and self.config.data.rollout_batch_size % self.config.worker.critic.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by global batch size.")

        self._create_dataloader()

    def _create_dataloader(self) -> None:
        train_dataset_attr = get_dataset_list(self.config.data.train_files, DATASET_INFO)
        val_dataset_attr = get_dataset_list(self.config.data.val_files,DATASET_INFO)
        train_datasets = []
        val_datasets = []
        for d_attr in train_dataset_attr:
            train_datasets.append(RLHFDataset(
            data_attr=d_attr, 
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            system_prompt=self.config.data.system_prompt,
            min_pixels=self.config.data.min_pixels,
            max_pixels=self.config.data.max_pixels,
            ))
        self.train_dataset = merge_datasets(train_datasets)
        # self.train_dataset = RLHFDataset(
        #     data_path=self.config.data.train_files,
        #     tokenizer=self.tokenizer,
        #     processor=self.processor,
        #     prompt_key=self.config.data.prompt_key,
        #     answer_key=self.config.data.answer_key,
        #     image_key=self.config.data.image_key,
        #     max_prompt_length=self.config.data.max_prompt_length,
        #     truncation="right",
        #     system_prompt=self.config.data.system_prompt,
        #     min_pixels=self.config.data.min_pixels,
        #     max_pixels=self.config.data.max_pixels,
        #     # column_mapping=self.config.column_mapping,
        # )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            sampler=sampler,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )
        for d_attr in val_dataset_attr:
            val_datasets.append(RLHFDataset(
            data_attr=d_attr, 
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            system_prompt=self.config.data.system_prompt,
            min_pixels=self.config.data.min_pixels,
            max_pixels=self.config.data.max_pixels,
            ))
        self.val_dataset = merge_datasets(val_datasets)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) == 1
        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")
        if self.config.trainer.max_steps is not None:
            training_steps = self.config.trainer.max_steps
        else:
            training_steps = len(self.train_dataloader) * self.config.trainer.total_episodes

        self.training_steps = training_steps
        self.config.worker.actor.optim.training_steps = training_steps
        self.config.worker.critic.optim.training_steps = training_steps
        print(f"Total training steps: {self.training_steps}")

    def _maybe_log_val_generations(self, inputs: List[str], outputs: List[str], scores: List[float]) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.val_generations_logger.log(self.config.trainer.logger, samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_scores = [], [], []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if "multi_modal_inputs" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = {"do_sample": False}
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        return {"val/test_score": reward_score}

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        print("init_workers: resource pool done")
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        print("init_workers: rool to cls done")
        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
            print("create actor and rollout done")
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls
            print("use critic done")
        print("Not use critic")
        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls
            print("use reference policy done")
        print("Reference policy done")

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls
            print("use reward model done")
        print("reward model done")

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)
        print("After resource_pool done")

        if self.use_critic:
            print("use critic")
            self.critic_wg: FSDPWorker = all_wg["critic"]
            self.critic_wg.init_model()
        print("critic---")
        if self.use_reference_policy:
            print("use reference policy")
            self.ref_policy_wg: FSDPWorker = all_wg["ref"]
            self.ref_policy_wg.init_model()
        print("reference policy---")

        if self.use_reward_model:
            print("use reward model")
            self.rm_wg: FSDPWorker = all_wg["rm"]
            self.rm_wg.init_model()
        print("reward model---")

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg: FSDPWorker = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/actor
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")

        self.actor_rollout_wg.save_checkpoint(
            actor_path,
            self.global_step,
            remove_previous_ckpt=self.config.trainer.remove_previous_ckpt,
        )

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(
                critic_path,
                self.global_step,
                remove_previous_ckpt=self.config.trainer.remove_previous_ckpt,
            )

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, "latest_global_step.txt")
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, remove_ckpt_after_load=self.config.trainer.remove_ckpt_after_load
        )
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(
                critic_path, remove_ckpt_after_load=self.config.trainer.remove_ckpt_after_load
            )

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def dynamic_reward_fn(self, batch,multi_modal_data, steps = 100000):
        rewards = []
        answers = []
        processes = []
        for index,sample in enumerate(batch):
            # print(f"problem_type = {sample.non_tensor_batch['problem_type']}") # OCR
            selected_reward_fn = self.reward_fn_dict.get(sample.non_tensor_batch.get("problem_type","None"), self.reward_fn_dict["numerical"])  # 默认用 "math"
            mm_data = multi_modal_data[index]["image"][0] if multi_modal_data is not None else None
            reward_tensor, answer_tensor, process_tensor = selected_reward_fn(sample, mm_data, steps) 
            if isinstance(reward_tensor, list):  
                reward_tensor = torch.tensor(reward_tensor, dtype=torch.float32)  # 确保是 tensor
            if isinstance(answer_tensor, list):  
                answer_tensor = torch.tensor(answer_tensor, dtype=torch.float32)
            if isinstance(process_tensor, list):  
                process_tensor = torch.tensor(process_tensor, dtype=torch.float32)
            reward_tensor = reward_tensor.squeeze(0) 
            answer_tensor = answer_tensor.squeeze(0)
            process_tensor = process_tensor.squeeze(0)
            rewards.append(reward_tensor)
            answers.append(answer_tensor)
            processes.append(process_tensor)
        rewards = torch.stack(rewards)
        answers = torch.stack(answers)
        processes = torch.stack(processes)
        return rewards, answers, processes
    
    def rank_dynamic_reward_fn(self, batch, multi_modal_data, n = 8, steps=100000):
        # multi_modal_data 是一个没有翻倍过的dict，uid->multimodal data
        rewards = [None] * len(batch)
        answer_rewards = [None] * len(batch)
        process_rewards = [None] * len(batch)
        gamma_rewards = [None] * len(batch)
        assert len(batch) % n == 0, "Batch size must be divisible by group_size"
        # 记录每个 sample 的位置
        uid_pos_map = {}  # uid -> list of (index, sample)
        for idx, sample in enumerate(batch):
            uid = sample.non_tensor_batch["uid"]
            if uid not in uid_pos_map:
                uid_pos_map[uid] = []
            uid_pos_map[uid].append((idx, sample))

        for uid, idx_sample_list in uid_pos_map.items():
            assert len(idx_sample_list) == n, f"UID {uid} does not have {n} samples"

            indices, samples = zip(*idx_sample_list)  # unzip
            group = collate_fn_protocal(samples)

            mm_group = multi_modal_data[uid] if multi_modal_data is not None else None

            problem_types = [s.non_tensor_batch.get("problem_type", "None") for s in samples]
            assert all(pt == problem_types[0] for pt in problem_types), \
                f"Not all problem_types are the same in UID {uid}: {problem_types}"
            problem_type = problem_types[0]
            
            selected_reward_fn = self.reward_fn_dict.get(problem_type, self.reward_fn_dict["numerical"])

            group_rewards, answer_tensor, process_tensor, gamma_tensor = selected_reward_fn(group, mm_group, steps, n)
            assert group_rewards.shape[0] == n
            # 确保是 tensor
            if isinstance(group_rewards, list):
                group_rewards = torch.tensor(group_rewards, dtype=torch.float32)
            if isinstance(answer_tensor, list):
                answer_tensor = torch.tensor(answer_tensor, dtype=torch.float32)
            if isinstance(process_tensor, list):
                process_tensor = torch.tensor(process_tensor, dtype=torch.float32)
            if isinstance(gamma_tensor, list):
                gamma_tensor = torch.tensor(gamma_tensor, dtype=torch.float32)

            group_rewards = group_rewards.squeeze()
            answer_tensor = answer_tensor.squeeze()
            process_tensor = process_tensor.squeeze()
            gamma_tensor = gamma_tensor.squeeze()

            # 把组内每个结果按照原 batch 的 index 放回正确位置
            for i, idx in enumerate(indices):
                rewards[idx] = group_rewards[i]
                answer_rewards[idx] = answer_tensor[i]
                process_rewards[idx] = process_tensor[i]
                gamma_rewards[idx] = gamma_tensor[i]

        rewards = torch.stack(rewards)
        answer_rewards = torch.stack(answer_rewards)
        process_rewards = torch.stack(process_rewards) 
        gamma_rewards = torch.stack(gamma_rewards)
        return rewards, answer_rewards, process_rewards, gamma_rewards


    # def dynamic_reward_fn(self, batch, multi_modal_data, steps=100000, max_workers=8):
    #     def compute_reward(index, sample):
    #         try:
    #             selected_reward_fn = self.reward_fn_dict.get(
    #                 sample.non_tensor_batch.get("problem_type", "None"),
    #                 self.reward_fn_dict["numerical"]
    #             )

    #             # 取出对应的 multimodal 信息
    #             mm_data = multi_modal_data[index]["image"][0] if multi_modal_data is not None else None

    #             # 将 multimodal data 一起传入 reward_fn，传入的是一个PIL.Image.Image，我们默认只有一张图片
    #             reward_tensor = selected_reward_fn(sample, mm_data, steps=steps)

    #             if isinstance(reward_tensor, list):
    #                 reward_tensor = torch.tensor(reward_tensor, dtype=torch.float32)
    #             return index, reward_tensor.squeeze(0)
    #         except Exception as e:
    #             print(f"[Error] computing reward for sample at index {index}: {e}")
    #             return index, torch.tensor(0.0)

    #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         futures = [executor.submit(compute_reward, i, s) for i, s in enumerate(batch)]

    #         rewards = [None] * len(batch)
    #         for future in as_completed(futures):
    #             index, reward = future.result()
    #             rewards[index] = reward

    #     return torch.stack(rewards)
    
    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=self.config.to_dict(),
        )
        val_metrics: Optional[Dict[str, Any]] = None
        self.global_step = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            print(f"Initial validation metrics: {val_metrics}.")
            logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        for _ in range(self.config.trainer.total_episodes):
            print(f"-----------------------epoch {_}------------------------------------")
            # for batch_dict in self.train_dataloader:
            for batch_dict in tqdm(self.train_dataloader, desc="Training", total=len(self.train_dataloader)):
                self.global_step += 1
                if self.global_step > self.training_steps:
                    break

                metrics, timing_raw = {}, {}
                batch: DataProto = DataProto.from_single_dict(batch_dict) # batch.non_tensor_batch.raw_prompt_ids # 这里的input_ids中有324个image token

                # pop those keys for generation
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    retained_multi_modal_data = batch.non_tensor_batch["multi_modal_data"]
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                    )
                else:
                    retained_multi_modal_data = None
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                # print(f"***** gen_batch = *****\n{gen_batch}")
                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):  # wg: worker group
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    if self.config.algorithm.adv_estimator == "remax":
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            # reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = self.dynamic_reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    assert len(batch) == len(retained_multi_modal_data)
                    retained_multi_modal_data = {
                        batch.non_tensor_batch["uid"][i]: retained_multi_modal_data[i]
                        for i in range(len(retained_multi_modal_data))
                    }
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                    # retained_multi_modal_data = repeat_multimodal_list(retained_multi_modal_data, repeat_times=self.config.worker.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        # input_ids1 = batch[0].batch["input_ids"]
                        # print(f"forward_micro_batch input_ids1: &&&&&&&{torch.sum(input_ids1 == 151655).item()}")
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)# 这里batch输入的也是324个
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # reward_fn should combine the results from reward model and rule-based results
                        if self.use_reward_model:
                            raise NotImplementedError("RM is not supported for PPO yet.")

                        # we combine with rule-based rm
                        # reward_tensor, answer_tensor, process_tensor = self.dynamic_reward_fn(batch, retained_multi_modal_data,steps=self.global_step) # batch size = 1024, mm_data size = 128
                        reward_tensor, answer_tensor, process_tensor, gamma_tensor = self.rank_dynamic_reward_fn(batch, retained_multi_modal_data, n=self.config.worker.rollout.n, steps=self.global_step)
                        batch.batch["token_level_scores"] = reward_tensor
                        batch.batch["answer_tensor"] = answer_tensor
                        batch.batch["process_tensor"] = process_tensor
                        batch.batch["gamma_tensor"] = gamma_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.worker.actor.use_kl_loss:  # not grpo's kl loss
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                            batch.batch["answer_tensor"] = batch.batch["answer_tensor"]
                            batch.batch["process_tensor"] = batch.batch["process_tensor"]
                            batch.batch["gamma_tensor"] = batch.batch["gamma_tensor"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)

                    # update actor
                    if self.config.trainer.critic_warmup <= self.global_step:
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)

                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        metrics.update(actor_metrics)

                    # validate
                    if ( # 目前没使用
                        self.val_reward_fn is not None
                        and self.config.trainer.val_freq > 0
                        and self.global_step % self.config.trainer.val_freq == 0
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics = self._validate()

                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_step)

        # # perform validation after training
        # if self.val_reward_fn is not None:
        #     if val_metrics is None or self.global_step % self.config.trainer.val_freq != 0:
        #         val_metrics = self._validate()
        #         logger.log(data=val_metrics, step=self.global_step)

        #     print(f"Final validation metrics: {val_metrics}.")
