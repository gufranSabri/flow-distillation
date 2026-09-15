#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --account=aip-lsigal
#SBATCH --mail-user=ahmed.ghuf@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=12:00:00

# SUBMIT THIS FROM THE REPO ROOT:
#   sbatch benchmark.slurm
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

# Base model
python benchmark.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --work-dir ./work_dir/benchmark_base_$(date +%d_%H_%M_%S)

python benchmark.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --work-dir ./work_dir/benchmark_base_$(date +%d_%H_%M_%S)

# Distilled model
#python benchmark.py \
#    --model <folder>/word_level_final \
#    --work-dir ./work_dir/benchmark_distilled_$(date +%d_%H_%M_%S)
