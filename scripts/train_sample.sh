set -euo pipefail
mkdir -p slurm work_dir

CONFIG="configs/word_level.yaml"
TEACHER="Qwen/Qwen2.5-3B-Instruct"
STUDENT="Qwen/Qwen2.5-1.5B-Instruct"

# Pass a run name as $1 to reuse an existing run; resolved under WORK_DIR_ROOT by main.py.
RUN_NAME="${1:-word_level_$(date +%d_%H_%M_%S)}"
WORK_DIR="$(python -c "import os,yaml;print(os.path.expanduser(yaml.safe_load(open('configs/common.yaml'))['WORK_DIR_ROOT']))")/${RUN_NAME}"

# ── 1. Train ───────────────────────────────────────────────────────────────────
python main.py --work-dir "$RUN_NAME" --config "$CONFIG"

# ── 2. Benchmark the distilled student ─────────────────────────────────────────
# Same script handles a hub id or a local checkpoint; pass --tasks to run a subset of
# MiniLLM's eval suite (default: dolly,self_inst,vicuna,s_ni,u_inst).
python benchmark.py \
    --model "${WORK_DIR}/word_level_final" \
    --work-dir "${WORK_DIR}/benchmark_student"

# ── 3. Baselines, for the comparison phase 1 is actually about ─────────────────
# The undistilled student is the number the distilled student has to beat; the teacher is
# the ceiling. Comment these out on reruns once you have them.
python benchmark.py \
    --model "$STUDENT" \
    --work-dir "${WORK_DIR}/benchmark_baseline_student"

python benchmark.py \
    --model "$TEACHER" \
    --work-dir "${WORK_DIR}/benchmark_teacher"

echo "Done. Results under ${WORK_DIR}/benchmark_*/summary.txt"
