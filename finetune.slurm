#!/bin/bash
#SBATCH --job-name=finetuning
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --account=aip-lsigal
#SBATCH --mail-user=ahmed.ghuf@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=12:00:00

# SUBMIT THIS FROM THE REPO ROOT:  sbatch finetune.slurm
set -euo pipefail
mkdir -p slurm work_dir

module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
bash ./scripts/install.sh

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1
export TF_CPP_MIN_LOG_LEVEL=3
export HF_HOME=/home/ahmedubc/scratch/hf_cache

# SFT baselines (no KD): the student and teacher, each fine-tuned directly on Dolly
# ground truth, cached to compare distill.py's KD-only student against.
MODELS=(
    # "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-3B"
)

for MODEL in "${MODELS[@]}"; do
    SAFE_NAME="${MODEL//\//_}"

    # Saved under WORK_DIR_ROOT/finetuned/<model> (configs/common.yaml resolves
    # --work-dir under WORK_DIR_ROOT unless given an explicit path).
    python finetune.py \
        --model "$MODEL" \
        --work-dir "finetuned/$SAFE_NAME" \
        --approach lora_ft \
        --TRAIN_EPOCHS 3

    # Benchmarking output stays in the repo's own work_dir, not WORK_DIR_ROOT.
    python benchmark.py \
        --model ~/scratch/distillation/finetuned/"$SAFE_NAME"/lora_ft_final \
        --work-dir ./work_dir/"$SAFE_NAME"
done
