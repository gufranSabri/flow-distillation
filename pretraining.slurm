#!/bin/bash
#SBATCH --job-name=pretraining
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --account=aip-lsigal
#SBATCH --mail-user=ahmed.ghuf@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00

# SUBMIT THIS FROM THE REPO ROOT:  sbatch pretraining.slurm
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
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-3B"
    "Qwen/Qwen2.5-7B"
)

for MODEL in "${MODELS[@]}"; do
    SAFE_NAME="${MODEL//\//_}"

    # Saved under WORK_DIR_ROOT/pretrained/<model> (configs/common.yaml resolves
    # --work-dir under WORK_DIR_ROOT unless given an explicit path).
    python pretraining.py \
        --model "$MODEL" \
        --work-dir "pretrained/$SAFE_NAME" \
        --config configs/pretrain.yaml \
        --epochs 3

    # Benchmarking output stays in the repo's own work_dir, not WORK_DIR_ROOT.
    python benchmark.py \
        --model ~/scratch/distillation/pretrained/"$SAFE_NAME"/pretrain_final \
        --work-dir ./work_dir/"$SAFE_NAME"
done
