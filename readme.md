# Antidistillation Sampling

This repository implements the techniques described in the paper [Antidistillation Sampling](https://antidistillation.com) for protecting language models from distillation attacks while preserving their utility.

## Installation

1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
2. Run `uv sync` to install dependencies
3. Install flash-attention: `uv add flash-attn --no-build-isolation`

## Quick Start

```bash
bash pipeline.sh
```

This runs the complete antidistillation sampling pipeline with default settings.

To run the pipelines for the `hendrycks_math` dataset, use `bash pipeline_math.sh`, and to run the pipeline for the `mmlu` dataset, use `bash pipeline_mmlu.sh`.
The hyperparameters and config settings in this project are optimized for running each experiment on a single 8xH100 node.

## What is Antidistillation Sampling?

[Antidistillation sampling](https://antidistillation.com) is a defense technique that protects language models from being copied through distillation. When models generate reasoning traces, these traces can be used to train "student" models that mimic the original—essentially stealing the model's capabilities. Antidistillation sampling modifies how the model generates reasoning traces to "poison" them, making these traces less useful for training copycat models while maintaining the original model's performance.

## How the Pipeline Works

The pipeline orchestrates a complete experiment to demonstrate antidistillation sampling:

1. **Generate holdout traces** - Creates reasoning traces from the teacher model on held-out data
2. **Compute proxy student gradients** - Calculates gradients needed for antidistillation sampling
3. **Sweep hyperparameters** - For each tau/lam/eps combination:
   - Generate poisoned training traces with antidistillation sampling
   - Train student models on the poisoned traces
   - Evaluate both student and teacher performance

### Pipeline Stages

When you run `pipeline.sh`, it executes these stages:

#### Stage 1: Holdout Trace Generation
- Uses the teacher model to generate reasoning traces on held-out data from the downstream task
- These holdout traces will be used to compute gradients for the proxy student model
- Uses `gentraces.py` with holdout data configuration

#### Stage 2: Proxy Student Gradient Computation
- Runs `save_grad.py` to compute gradients from a proxy student model using the holdout traces
- These gradients are used for antidistillation sampling to modify the teacher's outputs
- Saves gradients for use in subsequent stages

#### Stages 3-6: Hyperparameter Sweep
The pipeline tests multiple combinations of hyperparameters that trade off nominal teacher utility for antidistillation strength.

For each (tau, lam, eps) trio:

- **Stage 3**: Generate training traces with antidistillation sampling
  - Uses `gentraces.py` to create poisoned reasoning traces
  - tau: temperature for sampling
  - lam: strength of antidistillation (0 = no defense)
  - eps: precision parameter for finite difference approximation
  
- **Stage 4**: Distill student models
  - Runs `distill.py` to train student models on the poisoned traces using SFT
  - Uses LoRA for improved distillation performance
  - Tests whether the student can learn from poisoned data
  
- **Stage 5**: Evaluate student performance
  - Uses `gentraces.py` to test the distilled student model
  - Measures how well the student learned despite poisoning
  
- **Stage 6**: Evaluate teacher performance
  - Uses `gentraces.py` to test the teacher with the same antidistillation parameters
  - Verifies the teacher maintains quality despite poisoning

Note: `gentraces.py` is reused throughout the pipeline with different configurations for generating holdout traces, creating poisoned/temperature-sampled training data, and evaluating model performance.

## Customizing the Pipeline

### Key Configuration Variables

Edit these in `pipeline.sh`:

```bash
seed=42                   # Random seed for reproducibility
dataset=gsm8k             # Dataset to use (gsm8k, hendrycks_math, mmlu) may need to edit other hyperparamaters to optimize GPU usage (see pipeline_math.sh and pipeline_mmlu.sh for details).
exp_dir="${EXP_DIR:-...}" # Output directory for results
```

### Model Selection

The pipeline uses three models:
- **Teacher**: The model to protect (default: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`).
- **Proxy Student**: Used to compute gradients for ADS. Note that this must be the same architecture as the teacher so they can use the same tokenizer (default: `Qwen/Qwen2.5-3B`).
- **Student**: The attacker's model attempting distillation. Different from the proxy student to show that ADS works even if the defender has no knowledge of the architecture used to perform distillation (default: `meta-llama/Llama-3.2-3B`).

### Hyperparameter Grid

The pipeline tests multiple defense configurations. Modify `grid.py` to change:
- Temperature values (tau)
- Antidistillation strengths (lam)
- Precision parameters (eps)

### Configuration Files

- **gen_config.yaml**: Controls trace generation
  - Model settings
  - Token limits
  - Batch sizes
  - Dataset parameters

- **train_config.yaml**: Controls distillation training
  - Learning rates
  - Training epochs
  - LoRA parameters
  - Optimizer settings

- **acc_config.yaml**: Accelerate settings for distributed training

### Key Scripts

- `grid.py` - Generates hyperparameter combinations for experiments
- `save_grad.py` - Computes and caches gradients from the proxy student model
- `gentraces.py` - Generates reasoning traces (used for holdout data, training data, and evaluation)
- `distill.py` - Trains student models using the generated traces
- `utils.py` - Dataset loading and utility functions