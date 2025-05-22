# SophiaVL-R1


<p align="center">
  <a href="#">[paper]</a> &nbsp;&nbsp;
  <a href="https://huggingface.co/bunny127/SophiaVL-R1-7B">[SophiaVL-R1-7B model]</a> &nbsp;&nbsp;
  <a href="https://huggingface.co/bunny127/SophiaVL-R1-Thinking-Reward-Model-3B">[Thinking Reward Model]</a>
</p>

<p align="center">
<a href="https://huggingface.co/datasets/bunny127/SophiaVL-R1-130k">[SophiaVL-R1-130k Dataset]</a> &nbsp;&nbsp;
<a href="https://huggingface.co/datasets/bunny127/SophiaVL-R1-Thinking-156k">[SophiaVL-R1-Thinking-156k Dataset]</a>
</p>


## About SophiaVL-R1

We introduce **SophiaVL-R1** to explore the R1 paradigm with thinking-level reward model in vision-language reasoning. We trian a **Thinking Reward Model** to give thinking level reward signals and introuce **Trust-GRPO**, a trustworthiness-aware extension of GRPO that assigns dynamic trustworthiness weights to thinking-level rewards based on reward comparisons between correct and incorrect responses.

We construct two datasets: **SophiaVL-R1-130k** for Trust-GRPO training and **SophiaVL-R1-Thinking-156k** for training of Thinking Reward Model.

Our SophiaVL-R1-7B model achieves strong performance on multiple multimodal reasoning benchmarks. SophiaVL-R1-7B outperforms LLaVA-OneVision-72B on MathVista and MMMU, despite having 10× fewer parameters.

The overview of our Trust-GRPO method:

![](images/overview.png)
Below is a reasoning example that illustrates both wrong and correct thinking trajectories leading to correct answer. Our Thinking Reward Model distinguishes between them and assigns corresponding thinking rewards.
![](images/demo.png)

## Reqirements
### Software Requirements
- Python 3.9+
- transformers>=4.51.0
- flash-attn>=2.4.3
- vllm>=0.8.3

Start with the following commands:
```bash
git clone https://github.com/kxfan2002/SophiaVL-R1.git
cd SophiaVL-R1  
conda create -n sophiavl python=3.10
conda activate sophiavl
pip install -r requirements.txt
```

## Quick Start

### Download the model
We recommend using huggingface-cli to download the model. You can use the following command to download the model:
```bash
# download huggingface-cli
pip install -U huggingface_hub
huggingface-cli login

huggingface-cli download bunny127/SophiaVL-R1-7B --local-dir <local_dir>
```

### Dataset
We provide the [SophiaVL-R1-130k Dataset](https://huggingface.co/datasets/bunny127/SophiaVL-R1-130k) and the [SophiaVL-R1-Thinking-156k Dataset](https://huggingface.co/datasets/bunny127/SophiaVL-R1-Thinking-156k).

Download dataset:
```bash
# download huggingface-cli
pip install -U huggingface_hub
huggingface-cli login

huggingface-cli download bunny127/SophiaVL-R1-130k --repo-type dataset --local-dir <local_dir>
```

Our SophiaVL-R1-130k dataset is collected from publicly available datasets. Detail is demonstrated in figure below.

![](images/dataset.png)

#### Custom Dataset for Training
We support text-dataset and image-text dataset both in parquet and json file format. To train on your own datasets, please register your dataset in `verl/data/dataset_info.json` in the following format：
```python
"myDataset":{
        "file_path":"/path/to/your/dataset",
        "image_base_path":"/your/image/base/path",
        "columns":{
            "column_reponses_to_prompt":"prompt",
            "column_reponses_to_answer":"answer",
            "column_reponses_to_images":"images"
        }
    },
```

### Training

#### Training Scripts

Set these enviroment variables in `scripts/train_scripts/run_train.sh`:

- `OPENAI_API_KEY`: Key for Reward Model API
- `OPENAI_API_URL`: URL for Reward Model API
- `REWARD_MODEL`: Model name of Reward Model

Modify your training parameters in `scripts/train_scripts/fullsets.yaml` and start training with command:

```
bash scripts/train_scripts/run_train.sh
``` 

#### Merge Checkpoint in HuggingFace Format
The checkpoints saved during training need to be merged before using. This script will transfer the saved checkpoints to HuggingFace format. 

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

### Inference
We provide a simple inference script for you to test the model. The full script is [here](./scripts/inference_single.py). Have a try with your data!
```bash
# Modify the below fields to your test data
MODEL_PATH = "bunny127/SophiaVL-R1-7B" # or your local path
image_path = "/path/to/dataset/Math/CLEVR-Math/images/CLEVR_train_036427.png" # your local image path
prompt = "Subtract 0 cyan cubes. How many objects are left?"
question_type = "numerical" # numerical, multiple_choice, free-form, OCR
```
### Evaluation

We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for evaluation of SophiaVL-R1. To register our model in VLMEvalKit, add model description in `vlmeval/config.py`:

```python
"trained_model": partial(
        Qwen2VLChat,
        model_path="/path/to/model",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
```

## Performance of SophiaVL-R1-7B
SophiaVL-R1-7B demonstrates strong performance across multiple vision-language reasoning benchmarks, including both mathematical reasoning and general capability tasks. The tables below summarizes the results of SophiaVL-R1-7B compared to other models on these benchmarks.

![](images/table1.png)
![](images/table2.png)

### Training Curves
This shows the average outcome reward, which reflects the accuracy of final answers. As shown in the figure below, SophiaVL-R1 trained with Trust-GRPO achieves higher rewards with fewer training steps.
![](images/curve.png)

## More Reasoning Examples of SophiaVL-R1

![](images/exp1.png)
![](images/exp2.png)
![](images/exp3.png)
![](images/exp4.png)

## Acknowledgements

We sincerely appreciate the contributions of the open-source community. This work is built upon [EasyR1](https://github.com/hiyouga/EasyR1).