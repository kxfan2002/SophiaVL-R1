# SophiaVL-R1
## About SophiaVL-R1


## Reqirements
### Software Requirements
- Python 3.9+
- transformers>=4.51.0
- flash-attn>=2.4.3
- vllm>=0.8.3


## Quick Start


### Enviroment Variables

`OPENAI_API_KEY`: Key for Reward Model API
`OPENAI_API_URL`: URL for Reward Model API
`REWARD_MODEL`: Model name of Reward Model


### Scripts

Start training:
```
bash scripts/train_scripts/run_dsw.sh
```

Modify your training parameters in `scripts/train_scripts/fullsets.yaml`. `train_files` should be seperated with comma.

### Merge Checkpoint in Hugging Face Format
```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## Custom Dataset
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

## Acknowledgements

We sincerely appreciate the contributions of the open-source community. This work is built upon [EasyR1](https://github.com/hiyouga/EasyR1).