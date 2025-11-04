#!/bin/bash
# ================================================================================
# ANTIDISTILLATION SAMPLING PIPELINE
# ================================================================================
# This pipeline implements antidistillation sampling for protecting language models
# from distillation attacks (model stealing) while preserving model utility.
# 
# The process modifies the teacher model's sampling distribution to "poison" 
# reasoning traces, making them less useful for distillation without significantly
# degrading the teacher's performance. The pipeline includes:
# 1. Generating holdout traces using the teacher model
# 2. Computing the proxy student gradients for antidistillation sampling using the holdout traces
# 3. For each hyperparameter combination (tau, lam, eps):
#    - Generate poisoned reasoning traces with antidistillation sampling
#    - Distill student model from the teacher using the poisoned reasoning traces
#    - Evaluate the student model on the test set
#    - Evaluate the teacher model on the test set
# ================================================================================

set -e

# Color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
RESET='\033[0m'
YELLOW='\033[0;33m'
MAGENTA='\033[0;35m'

# Timer for tracking total pipeline execution time
SECONDS=0

# ================================================================================
# CONFIGURATION
# ================================================================================
seed=42                    # Random seed for reproducibility
dataset=gsm8k             # Dataset to use (GSM8K math problems)
exp_dir="${EXP_DIR:-$(git log -1 --pretty=%s | awk '{print $1}')}"  # Experiment directory
mkdir -p "${exp_dir}"
echo -e "${YELLOW}Experiment directory: ${exp_dir}${RESET}"

# Accelerate launch command with GPU configuration
PY="time uv run accelerate launch --config_file acc_config.yaml"

# ================================================================================
# HYPERPARAMETER GRID GENERATION
# ================================================================================
# Generate hyperparameter combinations (tau, lam, eps) for adversarial sampling
# tau: temperature parameter for sampling
# lam: lambda parameter for adversarial loss weighting  
# eps: epsilon parameter for gradient perturbation
python grid.py $(hostname) > params_temp.txt  # Distribute grid generation across multiple machines (identified by hostname)

# Initialize array to store hyperparameter combinations
declare -a taulamepss

# Read hyperparameter combinations into array
while IFS= read -r line; do
    taulamepss+=("$line")
done < params_temp.txt

echo -e "$(hostname)" 
echo -e "TAU      LAM      EPS" 
for item in "${taulamepss[@]}"; do
	echo $item
done

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ================================================================================
# MODEL CONFIGURATION
# ================================================================================
teacher="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"     # Large teacher model
proxy_student="Qwen/Qwen2.5-3B"                       # Proxy student for gradient computation (same architecture as teacher so we can use the same tokenizer)
student="meta-llama/Llama-3.2-3B"                     # Target student model attempting to distill from the teacher (different architecture to show that ADS works even when the student and teacher have different architectures)

# ================================================================================
# STAGE EXECUTION HELPER FUNCTION
# ================================================================================
# Runs a pipeline stage if its sentinel file doesn't exist (for resumability)
# Args: stage_name, sentinel_file, command_to_run
run_stage() {
    local stage="$1"
    local sentinel="$2"
    local cmd="$3"
    
    if [ -e "${sentinel}" ]; then
        echo -e "${YELLOW}⏭️ Skipping ${stage}: ${sentinel} already exists.${RESET}"
        return 0
    else
        local clean_cmd=$(echo "$cmd" | tr '\n' ' ' | sed 's/  */ /g')
        echo -e "${CYAN}🌀 ${stage}:\n> ${clean_cmd}${RESET}"
        eval "$cmd"
        echo -e "${GREEN}✅ ${stage} completed.${RESET}"
        return 0
    fi
}

# ================================================================================
# STAGE 1: HOLDOUT DATA GENERATION
# ================================================================================
# Generate traces on holdout data using the teacher model
# This is used to calculate the proxy student gradients for antidistillation sampling in stage 2
stage="HOLDOUT"
trace_name="holdout"
holdout_sentinel="${exp_dir}/traces/${trace_name}"
cmd="$PY \
    gentraces.py \
    hydra.run.dir=${exp_dir}/metadata/holdout \
    teacher=${teacher} \
    exp_dir=${exp_dir} \
    seed=${seed} \
    data_split=${dataset}_holdout \
    trace_name=${trace_name}"
run_stage "$stage" "$holdout_sentinel" "$cmd"

# ================================================================================
# STAGE 2: STUDENT GRADIENT COMPUTATION
# ================================================================================
# Compute and save gradients from the proxy student model using the holdout traces
# These gradients are used for antidistillation sampling in stage 3
stage="STUDENT GRAD"
grad_path="${exp_dir}/student_grads.pt"
grad_sentinel="${grad_path}"
cmd="$PY save_grad.py ${holdout_sentinel}.yaml --proxy_student=${proxy_student}"
run_stage "$stage" "$grad_sentinel" "$cmd"

# ================================================================================
# MAIN PIPELINE LOOP: ANTIDISTILLATION SAMPLING FOR EACH HYPERPARAMETER SET
# ================================================================================
for taulameps in "${taulamepss[@]}"; do
    read -r tau lam eps <<< "$taulameps"

    # ============================================================================
    # STAGE 3: ANTIDISTILLATION SAMPLING
    # ============================================================================
    # Generate training traces using antidistillation sampling with current hyperparameters
    # - tau: controls sampling temperature
    # - lam: weights antidistillation loss (0.0 = no antidistillation sampling)
    # - eps: finite difference approximation precision
    stage="ADS SAMPLING TAU=${tau}, LAM=${lam}, EPS=${eps}"
    trace_name="tau${tau}_lam${lam}_eps${eps}"
    ad_sentinel="${exp_dir}/traces/${trace_name}"
    # Use larger batch size when no antidistillation sampling (lam=0.0)
    batch_size=$([[ "$lam" == "0.0e+00" ]] && echo "512" || echo "192")
    cmd="$PY \
        gentraces.py \
        hydra.run.dir=${exp_dir}/metadata/train/${trace_name} \
        teacher=${teacher} \
        proxy_student=${proxy_student} \
        exp_dir=${exp_dir} \
        seed=${seed} \
        data_split=${dataset}_train \
        grad_path=${grad_path} \
        batch_size=${batch_size} \
        tau=${tau} \
        lam=${lam} \
        eps=${eps} \
        trace_name=${trace_name}"
    run_stage "$stage" "$ad_sentinel" "$cmd"

    # ============================================================================
    # STAGE 4: STUDENT MODEL DISTILLATION
    # ============================================================================
    # Train student model using traces generated with antidistillation sampling
    # The student model tries to mimic the teacher model's reasoning process using SFT
    stage="DISTILLATION TAU=${tau}, LAM=${lam}, EPS=${eps}"
    model_name="student_tau${tau}_lam${lam}_eps${eps}"
    model_path="${exp_dir}/models/${model_name}"
    ad_traces="${exp_dir}/traces/tau${tau}_lam${lam}_eps${eps}"
    distill_sentinel="${model_path}/final"
    cmd="$PY \
        distill.py \
        hydra.run.dir=${exp_dir}/metadata/distill/${model_name} \
        student=${student} \
        tokenizer=${student}-Instruct \
        exp_dir=${exp_dir} \
        train_traces=${ad_traces} \
        holdout_traces=${holdout_sentinel} \
        model_name=${model_name}"
    run_stage "$stage" "$distill_sentinel" "$cmd"

    # ============================================================================
    # STAGE 5: STUDENT MODEL EVALUATION
    # ============================================================================
    # Evaluate the distilled student model on test data
    # Measures how well the student performs on unseen examples
    stage="EVAL STUDENT TAU=${tau}, LAM=${lam}, EPS=${eps}"
    eval_traces="eval_student_tau${tau}_lam${lam}_eps${eps}"
    eval_sentinel="${exp_dir}/traces/${eval_traces}"
    cmd="$PY \
        gentraces.py \
        hydra.run.dir=${exp_dir}/metadata/eval/tau${tau}_lam${lam}_eps${eps} \
        teacher=${distill_sentinel} \
        teacher_cfg=${model_path}.yaml \
        use_wandb=true \
        is_teacher=false \
        exp_dir=${exp_dir} \
        seed=${seed} \
        data_split=${dataset}_test \
        trace_name=${eval_traces}"
    run_stage "$stage" "$eval_sentinel" "$cmd"

    # ============================================================================
    # STAGE 6: TEACHER MODEL EVALUATION
    # ============================================================================
    # Evaluate the original teacher model on test data with same antidistillation parameters
    # Provides baseline performance comparison and measures the degradation in teacher model nominal utility due to ADS
    stage="EVAL TEACHER TAU=${tau}, LAM=${lam}, EPS=${eps}"
    eval_traces="eval_teacher_tau${tau}_lam${lam}_eps${eps}"
    eval_sentinel="${exp_dir}/traces/${eval_traces}"
    cmd="$PY \
        gentraces.py \
        hydra.run.dir=${exp_dir}/metadata/eval/tau${tau}_lam${lam}_eps${eps} \
        teacher=${teacher} \
        teacher_cfg=${model_path}.yaml \
        use_wandb=true \
        is_teacher=true \
        proxy_student=${proxy_student} \
        exp_dir=${exp_dir} \
        seed=${seed} \
        data_split=${dataset}_test \
        grad_path=${grad_path} \
        batch_size=${batch_size} \
        tau=${tau} \
        lam=${lam} \
        eps=${eps} \
        trace_name=${eval_traces}"
    run_stage "$stage" "$eval_sentinel" "$cmd"
done

# ================================================================================
# PIPELINE COMPLETION
# ================================================================================
duration=$SECONDS
printf "${WHITE}\n🎯 All processes completed in %02dh:%02dm:%02ds${RESET}\n" \
  $((duration / 3600)) $(((duration % 3600) / 60)) $((duration % 60))
