# -*- coding: utf-8 -*-
# utils.py
import os
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import set_seed

ANSWER_FORCE_STRING = "\n\n**Final Answer**\n\\[\\boxed{"
SYSTEM_PROMPT = (
    "You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
    "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions.\n"
)

def init(user_name, seed=42, babel=False):
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    cuda_capability = torch.cuda.get_device_capability()
    if cuda_capability[0] >= 8:  # Ampere or newer
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_SDP_ATTENTION"] = "never"
    if babel:
        cache_dir = Path(f"/scratch/" + user_name + "/triton_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"  # Use loopback for GLOO
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"  # Use loopback for NCCL

def load_gsm8k(split="train"):
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.rename_columns({"question": "problem", "answer": "solution"})
    dataset.shuffle(seed=42)
    train_size = int(len(dataset) * 0.7)
    if split == "train":
        return dataset.select(range(train_size))
    elif split == "holdout":
        return dataset.select(range(train_size, len(dataset)))
    elif split == "test":
        dataset = load_dataset("madrylab/gsm8k-platinum", split="test")
        dataset = dataset.rename_columns({"question": "problem", "answer": "solution"})
        return dataset
    else:
        raise ValueError("split must be either 'train', 'test', or 'holdout'")

def load_hendrycks_math_dataset(split="train"):

    if split not in ["train", "test", "holdout"]:
        raise ValueError("split must be either 'train', 'test', or 'holdout'")
    ds_split = "test" if split == "test" else "train"
    subsets = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    datasets = [load_dataset('EleutherAI/hendrycks_math', s, split=ds_split) for s in subsets]
    dataset = concatenate_datasets(datasets)

    if ds_split == "test":
        return dataset

    dataset = dataset.shuffle(seed=42)
    train_size = int(len(dataset) * 0.7)
    if split == "train":
        dataset = dataset.select(range(train_size))
    elif split == "holdout":
        dataset = dataset.select(range(train_size, len(dataset)))
    return dataset

def load_mmlu(split="train"):
    if split in ["train", "holdout"]:
        ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    else:
        ds = load_dataset("cais/mmlu", "all", split="test")
    ds.shuffle(seed=42)
    train_size = int(len(ds) * 0.7)
    if split == "train":
        ret = ds.select(range(train_size))
    elif split == "holdout":
        ret = ds.select(range(train_size, len(ds)))
    elif split == "test":
        ret = ds
    else:
        raise ValueError("split must be train, test, or holdout")
    
    def to_math_format(mmlu_ds):
        def format_example(ex):
            choices = ex['choices']
            prompt = f"{ex['question']}\n"
            prompt += '\n'.join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            return prompt

        def transform(ex):
            problem = format_example(ex)
            letter = chr(65 + ex['answer'])
            sol = "\\boxed{" + letter + "}"
            return {'problem': problem, 'solution': sol}

        return mmlu_ds.map(transform, remove_columns=mmlu_ds.column_names)

    return to_math_format(ret)