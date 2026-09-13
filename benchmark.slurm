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
#SBATCH --time=1-00:00:00

# SUBMIT THIS FROM THE REPO ROOT:
#   sbatch benchmark.slurm Qwen/Qwen2.5-1.5B-Instruct
#   sbatch benchmark.slurm ~/scratch/distillation/<run>/word_level_final [extra flags]
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
# humaneval/mbpp execute model-generated code
export HF_ALLOW_CODE_EVAL=1

MODEL="${1:?usage: sbatch benchmark.slurm <hub-id-or-checkpoint-path> [extra flags]}"
shift

python benchmark.py \
    --model "$MODEL" \
    --work-dir "./work_dir/benchmark_$(basename "$MODEL")_$(date +%d_%H_%M_%S)" "$@"


python benchmark.py --model ~/scratch/distillation/test/word_level_final --work-dir work_dir/test
