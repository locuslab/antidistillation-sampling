# -*- coding: utf-8 -*-
# ================================================================================
# GENTRACES.PY - ANTIDISTILLATION SAMPLING TRACE GENERATION
# ================================================================================
# This script generates reasoning traces from language models with optional
# antidistillation sampling (ADS) for protecting against model distillation attacks.
#
# Key functionality:
# 1. Generate clean reasoning traces (when lam=0)
# 2. Generate "poisoned" traces with ADS (when lam>0) that maintain teacher utility
#    but reduce effectiveness for student distillation
# 3. Evaluate model performance on reasoning tasks
# 4. Support for answer forcing to improve trace quality
#
# The antidistillation mechanism works by:
# - Using gradients from a proxy student model 
# - Modifying the teacher's sampling distribution via finite difference approximation
# - Sampling in directions that would hurt student performance if it learns from traces
# ================================================================================

import ast
import json
import logging
import os
import random
import shutil
import tempfile
from io import StringIO
from pathlib import Path

import datasets
import hydra
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import gather_object
from hydra.core.hydra_config import HydraConfig
from math_verify import parse, verify
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorWithPadding, LogitsProcessor,
                          LogitsProcessorList)
from transformers import logging as hf_logging

import wandb
from utils import (ANSWER_FORCE_STRING, SYSTEM_PROMPT, init, load_gsm8k,
                   load_hendrycks_math_dataset, load_mmlu)

# ================================================================================
# SETUP AND INITIALIZATION
# ================================================================================
accelerator = Accelerator()
log = logging.getLogger(__name__)

# Disable verbose logging for non-main processes to reduce noise
if not accelerator.is_main_process:
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    datasets.disable_progress_bar()
    tqdm = lambda x, *args, **kwargs: x

# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

def log_color(content, title=""):
    """Enhanced logging with colored console output for main process."""
    try:
        console = Console()
        console.print(Panel(content, title=title, border_style="cyan", title_align="left"))

        # Log the message as plain text for log files
        string_io = StringIO()
        plain_console = Console(file=string_io, highlight=False)
        plain_console.print(Panel(content, title=title, border_style="none", title_align="left"))
        log.info("\n" + string_io.getvalue())
    except Exception as e:
        # Fallback to plain text logging if Console fails
        log.info(f"Error logging content: {e}")

def is_correct(example, trace_colname):
    """
    Evaluate if a generated trace produces the correct answer for a math problem.
    
    Uses math_verify to parse and compare solutions. Handles cases where the 
    answer forcing string splits the response.
    """
    trace = example[trace_colname]
    try:
        soln = parse(example["solution"])
        if ANSWER_FORCE_STRING in trace:
            # Handle answer forcing: try multiple ways to extract the answer
            parts = trace.split(ANSWER_FORCE_STRING)
            alt_ans1 = ANSWER_FORCE_STRING.join(parts[:-1])
            alt_ans2 = parts[-1]
            res = any(verify(soln, parse(ans)) for ans in [trace, alt_ans1, alt_ans2])
        else:
            res = verify(soln, parse(trace))
    except:
        print(f"Error parsing trace: {trace} and comparing with solution: {example['solution']}")
        res = False
    return {"is_correct": res}

# ================================================================================
# CACHED MODEL WRAPPER FOR EFFICIENT INFERENCE
# ================================================================================

class CachedModelWrapper:
    """
    Wrapper for language models that implements KV-cache optimization.
    
    This avoids recomputing attention for previously processed tokens during
    incremental generation, which is crucial for the antidistillation sampling
    where we need to call the student models multiple times per token.
    """
    def __init__(self, model):
        self.model = model
        self.past_key_values = None
        self.last_position = 0

    def __call__(self, input_ids, attention_mask=None):
        # If no cache or sequence is shorter than last position, recompute from scratch
        if self.past_key_values is None or input_ids.shape[1] <= self.last_position:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
            self.past_key_values = outputs.past_key_values
            self.last_position = input_ids.shape[1]
            return outputs.logits

        # Use cached key-values and only process new tokens
        new_token = input_ids[:, -1:]
        outputs = self.model(
            new_token,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=self.past_key_values,
            return_dict=True
        )
        self.past_key_values = outputs.past_key_values
        self.last_position += 1
        return outputs.logits

# ================================================================================
# MAIN TRACE GENERATION FUNCTION
# ================================================================================

@hydra.main(config_path=".", config_name="gen_config", version_base="1.3")
def main(cfg: DictConfig):

    # ============================================================================
    # CONFIGURATION VALIDATION AND SETUP
    # ============================================================================
    cfg.antidistillation = cfg.lam != 0
    cfg.wandb_lam = 1e-8 if cfg.lam == 0 else cfg.lam  # Wandb doesn't allow log scale for 0 values
    
    # Validate antidistillation requirements
    if cfg.antidistillation:
        assert cfg.proxy_student is not None, "Proxy student model must be specified for antidistillation"
        assert cfg.grad_path is not None, "Grad path must be specified for antidistillation"

    if cfg.trace_name == "REPLACE_ME":
        raise ValueError("Trace name must be specified")

    # Initialize random seeds and logging
    init(os.getenv("USER"), cfg.seed)

    if accelerator.is_main_process:
        content = Syntax(OmegaConf.to_yaml(cfg, resolve=True), 'yaml', theme="monokai")
        log_color(content, title="Config")

    # ============================================================================
    # TOKENIZER SETUP
    # ============================================================================
    # Configure tokenizer with proper padding and special tokens for different model families
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer,
        use_fast=True,
        fast_tokenizer=True,
        trust_remote_code=True,
        padding_side="left",  # Left padding for generation
    )
    
    # Model-specific tokenizer configuration
    if "llama" in cfg.tokenizer.lower():
        eot_token_id = 128009
        eos_token_id = 128001
        tokenizer.pad_token_id = 128004
        tokenizer.eos_token_id = eos_token_id
        tokenizer.add_eos_token = False
        eos_token = tokenizer.eos_token
    else:
        eos_token = tokenizer.eos_token
        bos_token = tokenizer.bos_token or ""
        special_tokens = {"pad_token": "[PAD]"}
        tokenizer.add_special_tokens(special_tokens)

    # ============================================================================
    # TEACHER MODEL SETUP
    # ============================================================================
    # Load the teacher model that will generate the reasoning traces
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Use Flash Attention for efficiency
        torch_dtype=torch.bfloat16,
        use_cache=True,
    ).to(accelerator.device)
    teacher.generation_config.pad_token_id = tokenizer.pad_token_id
    teacher.resize_token_embeddings(len(tokenizer))

    # ============================================================================
    # ANTIDISTILLATION SETUP
    # ============================================================================
    # If antidistillation is enabled, set up proxy student models for gradient computation
    if cfg.antidistillation:
        # Load two copies of the proxy student model:
        # - student: perturbed with +eps * gradients
        # - dstudent: perturbed with -eps * gradients
        # This enables finite difference approximation for the antidistillation term
        student = CachedModelWrapper(AutoModelForCausalLM.from_pretrained(
            cfg.proxy_student,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
            use_cache=True,
        ).to(accelerator.device))
        
        dstudent = CachedModelWrapper(AutoModelForCausalLM.from_pretrained(
            cfg.proxy_student,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
            use_cache=True,
        ).to(accelerator.device))

        student.model.resize_token_embeddings(len(tokenizer))
        dstudent.model.resize_token_embeddings(len(tokenizer))

        # Load precomputed gradients and apply finite difference perturbations
        grads = torch.load(cfg.grad_path, map_location='cpu')
        if accelerator.is_main_process:
            log.info(f"Using eps: {cfg.eps}")
        
        # Apply +eps * gradient perturbation to student model
        used_grads = set()
        param_sq, grad_sq, num_params = 0, 0, 0
        for name, param in student.model.named_parameters():
            module_name = 'module.' + name
            if module_name in grads:
                grad = grads[module_name].to(param.device, dtype=torch.float32)
                param.data = (param.data.to(torch.float32) + cfg.eps * grad).to(param.data.dtype)
                param_sq += torch.sum(param.data.to(torch.float32) ** 2).item()
                grad_sq += torch.sum(grad ** 2).item()
                num_params += torch.numel(param.data)
                used_grads.add(module_name)
        
        assert used_grads == set(grads.keys()), f"Some gradients were not used or set: {set(grads.keys()) ^ used_grads}"
        if accelerator.is_main_process:
            log_color(f"{param_sq ** 0.5 / num_params ** 0.5:.2e}", title="Param RMSNorm")
            log_color(f"{grad_sq ** 0.5 / num_params ** 0.5:.2e}", title="Grad RMSNorm")

        # Apply -eps * gradient perturbation to dstudent model  
        used_grads = set()
        for name, param in dstudent.model.named_parameters():
            module_name = 'module.' + name
            if module_name in grads:
                grad = grads[module_name].to(param.device, dtype=torch.float32)
                param.data = (param.data.to(torch.float32) - cfg.eps * grad).to(param.data.dtype)
                used_grads.add(module_name)
        
        assert used_grads == set(grads.keys()), f"Some gradients were not used or set: {set(grads.keys()) ^ used_grads}"
        del grads
        if accelerator.is_main_process:
            log.info('Calculated grads')

    # ============================================================================
    # DATASET LOADING AND PREPROCESSING
    # ============================================================================
    # Load the appropriate dataset based on configuration
    if "gsm8k" in cfg.data_split:
        dataset = load_gsm8k(split=cfg.data_split.split("_")[1])
    elif "math" in cfg.data_split:
        dataset = load_hendrycks_math_dataset(split=cfg.data_split.split("_")[1])
    elif "mmlu" in cfg.data_split:
        dataset = load_mmlu(split=cfg.data_split.split("_")[1])
    else:
        raise ValueError(f"Unknown dataset and split: {cfg.data_split}")

    # Limit dataset size if specified
    if cfg.max_samples is not None:
        dataset = dataset.take(min(cfg.max_samples, len(dataset)))
    
    # Preprocess dataset on main process to convert problems to tokenized inputs
    if accelerator.is_main_process:
        def preprocess_function(examples):
            # Create chat messages with system prompt and user problem
            messages = [[{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": problem.strip() + "\n"}] for problem in examples["problem"]]
            # Apply chat template to get tokenized inputs
            tokens = [toks for toks in tokenizer.apply_chat_template(messages, add_generation_prompt=True)]
            seq_lengths = [len(toks) for toks in tokens]
            return {"input_ids": tokens, "seq_lengths": seq_lengths}

        proc_dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=96,
            desc="Preprocessing dataset",
            load_from_cache_file=True,
        )
        # Filter out sequences that are too long
        proc_dataset = proc_dataset.filter(lambda x: x["seq_lengths"] <= cfg.max_prompt_length)
        log_color(tokenizer.decode(proc_dataset[0]['input_ids']), title="Example Input")
        seq_length_stats = proc_dataset.to_pandas()["seq_lengths"].describe()
        log_color(str(seq_length_stats.round(2)), title="Sequence Lengths")
        proc_dataset = proc_dataset.remove_columns("seq_lengths")
        proc_dataset.save_to_disk("/tmp/cached_proc_dataset")
    
    # Synchronize processes and load cached dataset
    accelerator.wait_for_everyone()
    proc_dataset = datasets.load_from_disk("/tmp/cached_proc_dataset")

    # Shard dataset across multiple processes for distributed processing
    num_shards = accelerator.num_processes
    shard_id = accelerator.process_index
    dataset_shard = proc_dataset.shard(num_shards=num_shards, index=shard_id)
    ptds_shard = dataset_shard.remove_columns(dataset.column_names)

    # Create dataloader for batch processing
    dataloader = DataLoader(
        ptds_shard,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    # ============================================================================
    # ANTIDISTILLATION LOGITS PROCESSOR
    # ============================================================================
    
    class LogprobsModifier(LogitsProcessor):
        """
        Logits processor that implements antidistillation sampling.
        
        Modifies the teacher's token probability distribution by adding an
        antidistillation term computed via finite difference approximation
        using the perturbed proxy student models.
        
        The core formula is:
        new_logits = original_logits + (λ/(2ε)) * (student_logits - dstudent_logits)
        
        Where:
        - λ (lam): controls strength of antidistillation 
        - ε (eps): finite difference step size
        - student_logits: logits from +ε perturbed model
        - dstudent_logits: logits from -ε perturbed model
        """
        def __init__(self, lam, eps, attention_mask, repetition_penalty):
            super().__init__()
            self.lam = lam
            self.eps = eps
            self.attention_mask = attention_mask
            self.repetition_penalty = repetition_penalty

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            # Pad attention mask to match current sequence length
            attention_mask = F.pad(self.attention_mask, pad=(0, input_ids.shape[1]-self.attention_mask.shape[1]), value=1)
            
            # Get logits from both perturbed proxy student models
            out_target = student(input_ids=input_ids, attention_mask=attention_mask)[:, -1]
            out_Dtarget = dstudent(input_ids=input_ids, attention_mask=attention_mask)[:, -1]
            
            # Compute antidistillation term using finite difference approximation
            ad_term = (self.lam / (2*self.eps)) * (out_target.float() - out_Dtarget.float())

            # Add antidistillation term to original logits
            scores = scores.float() + ad_term
            
            return scores

    # ============================================================================
    # TRACE GENERATION LOOP
    # ============================================================================
    # Generate reasoning traces for each batch in the dataset
    traces = []
    for batch in tqdm(dataloader, total=len(dataloader), desc=f"tau={cfg.tau:.2e}, lam={cfg.lam:.2e}, eps={cfg.eps:.2e}"):
        batch = {key: value.to(accelerator.device) for key, value in batch.items()}
        with torch.inference_mode():
            outputs = teacher.generate(
                **batch,
                max_new_tokens=None,
                max_length=cfg.max_length,
                temperature=cfg.tau if cfg.tau > 0 else None,
                do_sample=True if cfg.tau > 0 else False,
                top_p=0.95 if cfg.tau > 0 else None,
                # Use antidistillation logits processor if enabled
                logits_processor=(
                    LogitsProcessorList([LogprobsModifier(cfg.lam, cfg.eps, batch["attention_mask"], cfg.repetition_penalty)])
                    if cfg.antidistillation else None
                ),
                renormalize_logits=True if cfg.antidistillation else False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0 if cfg.antidistillation else cfg.repetition_penalty,
            )
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            for text in generated_texts:
                text = text.replace(tokenizer.pad_token, "")
                traces.append(text)

    if accelerator.is_main_process:
        log_color(traces[0], title="First trace")
    
    # Add generated traces to dataset and clean up GPU memory
    dataset_shard = dataset_shard.add_column(cfg.trace_colname, traces)
    if cfg.antidistillation:
        del student, dstudent
        torch.cuda.empty_cache()

    # ============================================================================
    # CORRECTNESS EVALUATION
    # ============================================================================
    # Evaluate correctness of the raw generated traces
    dataset_shard = dataset_shard.map(
        is_correct,
        fn_kwargs={"trace_colname": cfg.trace_colname},
        desc="Checking raw correctness"
    )
    dataset_shard = dataset_shard.rename_columns({"is_correct": "is_raw_correct"})

    # ============================================================================
    # ANSWER FORCING (OPTIONAL)
    # ============================================================================
    # Answer forcing improves trace quality by explicitly prompting for a final answer
    # after the reasoning is complete
    
    # Detect the response string format based on model architecture
    response_strings = {
        "llama": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "qwen": "<|im_start|>assistant\n",
        "r1": "<｜Assistant｜>"
    }
    response_string = None
    for _, value in response_strings.items():
        if value in traces[0]:
            response_string = value
            break
    if response_string is None:
        raise ValueError("Response string not found in tokenizer chat template")
    
    if cfg.answer_force:
        # Prepare traces for answer forcing by adding the forcing string
        traces_ = []
        for text in traces:
            if "</think>" in text.split(response_string)[-1]:
                traces_.append(text + ANSWER_FORCE_STRING)
            else:
                traces_.append(text + "\n</think>" + ANSWER_FORCE_STRING)
        
        # Process in smaller batches to generate final answers
        af_batch_size = cfg.batch_size // 2
        traces_batched = [traces_[i:i+af_batch_size] for i in range(0, len(traces_), af_batch_size)]
        traces_af = []
        for batch in tqdm(traces_batched, total=len(traces_batched)):
            batch = [text.replace(bos_token, '', 1).replace(eos_token, '') for text in batch]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(accelerator.device)
            with torch.inference_mode():
                # Generate final answer without antidistillation (deterministic)
                outputs = teacher.generate(
                    **inputs,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    logits_processor=None,
                    renormalize_logits=False,
                    max_new_tokens=32,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id
                )
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                for text in generated_texts:
                    text = text.replace(tokenizer.pad_token, "")
                    if not text.endswith(tokenizer.eos_token):
                        text = text + tokenizer.eos_token
                    traces_af.append(text)

        if accelerator.is_main_process:
            log_color(traces_af[0], title="First af trace")
        del teacher
        torch.cuda.empty_cache()

        # Add answer-forced traces and evaluate their correctness
        dataset_shard = dataset_shard.add_column(cfg.trace_colname+"_af", traces_af)
        dataset_shard = dataset_shard.map(
            is_correct,
            fn_kwargs={"trace_colname": cfg.trace_colname+"_af"},
            desc="Checking af correctness"
        )
        dataset_shard = dataset_shard.rename_columns({"is_correct": "is_af_correct"})

    # ============================================================================
    # DATASET SAVING AND AGGREGATION
    # ============================================================================
    # Save dataset shards temporarily and then aggregate on main process
    tmp_dir = Path(tempfile.mkdtemp(prefix="tmp_ds_"))
    shard_path = tmp_dir / f"shard_rank_{accelerator.process_index:05d}"
    dataset_shard.save_to_disk(shard_path)

    accelerator.wait_for_everyone()

    # Gather all shard paths and concatenate datasets on main process
    all_paths = gather_object([shard_path])
    if accelerator.is_main_process:
        trace_dataset = datasets.concatenate_datasets([datasets.load_from_disk(path) for path in all_paths])

        # Save final dataset in multiple formats
        trace_dataset.save_to_disk(cfg.trace_path)
        trace_dataset.to_parquet(cfg.trace_path + ".parquet")

        # Clean up temporary files
        for path in all_paths:
            shutil.rmtree(path, ignore_errors=True)

        # ========================================================================
        # RESULTS LOGGING AND STATISTICS
        # ========================================================================
        # Display example results
        example_row = trace_dataset[random.randint(0, len(trace_dataset)-1)]
        log_color(example_row["problem"], title="Example Problem")
        log_color(example_row["solution"], title="Example Solution")
        log_color(example_row[cfg.trace_colname], title=f"Example Trace [tau={cfg.tau:.2e}, lam={cfg.lam:.2e}, eps={cfg.eps:.2e}]")
        if cfg.answer_force:
            log_color(example_row[cfg.trace_colname + "_af"], title=f"Example AF Trace [tau={cfg.tau:.2e}, lam={cfg.lam:.2e}, eps={cfg.eps:.2e}]")

        # Compute performance statistics
        trace_df = trace_dataset.to_pandas()
        trace_len_stats = {k:float(v) for k,v in trace_df[cfg.trace_colname].map(lambda x: len(tokenizer.encode(x))).describe().items()}
        raw_accuracy = float(trace_df["is_raw_correct"].mean())
        af_accuracy = float(trace_df["is_af_correct"].mean())

        # Prepare complete configuration with results
        full_cfg = OmegaConf.to_container(cfg, resolve=True)
        hydra_cfg = HydraConfig.get()
        full_cfg["hydra"] = {
            "run_dir": hydra_cfg.run.dir,
            "job_name": hydra_cfg.job.name,
            "cwd": hydra_cfg.runtime.cwd,
        }
        full_cfg["stats"] = {
            "raw_accuracy": raw_accuracy,
            "af_accuracy": af_accuracy,
            "trace_len_stats": trace_len_stats,
        }
        
        # Save configuration and results
        yaml_path = cfg.trace_path + ".yaml"
        with open(yaml_path, "w") as f:
            OmegaConf.save(full_cfg, f)
        log.info(f"Configuration saved to {yaml_path}")

        # Append metadata to registry for experiment tracking
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        with open(cfg.trace_registry, "a") as f:  # jsonl
            f.write(json.dumps(flatten_dict(full_cfg)) + "\n")
        log.info(f"Metadata appended to {cfg.trace_registry}")

        content = Syntax(OmegaConf.to_yaml(full_cfg, resolve=True), 'yaml', theme="monokai")
        log_color(content, title="Final Config")

        # ========================================================================
        # WANDB LOGGING (OPTIONAL)
        # ========================================================================
        # Log results to Weights & Biases if enabled
        if cfg.use_wandb and cfg.teacher_cfg:
            with open(cfg.teacher_cfg, "r") as f:
                teacher_cfg = yaml.safe_load(f)
            wandb_run_id = teacher_cfg.get("wandb_run_id")
            if wandb_run_id is None:
                raise ValueError("wandb is true but wandb_run_id not found in teacher config")
            wandb.init(
                project="antidistillation",
                id=wandb_run_id,
                resume="allow",
            )
            if cfg.is_teacher:
                wandb.log({
                    "teacher_raw_accuracy": raw_accuracy,
                    "teacher_af_accuracy": af_accuracy,
                })
            else:
                wandb.log({
                    "student_raw_accuracy": raw_accuracy,
                    "student_af_accuracy": af_accuracy,
                })
    
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()
