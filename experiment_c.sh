#!/bin/bash
#SBATCH --job-name=exp_c_%j
#SBATCH --time=0-4:00:00
#SBATCH --account=def-rgrosse
#SBATCH --mem=128G
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/exp_c_%x_%j.out
#SBATCH --error=slurm_logs/exp_c_%x_%j.err

# Usage:  sbatch --job-name=exp_c_velocity experiment_c.sh velocity
#    or:  sbatch experiment_c.sh velocity --blocks value_patching

EXPERIMENT=${1:-velocity}
EXTRA_ARGS=("${@:2}")

mkdir -p slurm_logs

module load python cuda scipy-stack arrow

source venv/bin/activate

echo "=========================================="
echo "Experiment C: ${EXPERIMENT}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node  : $(hostname)"
echo "GPUs  : $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo "Start : $(date)"
echo "=========================================="

python intervention_token_c.py --experiment "${EXPERIMENT}" "${EXTRA_ARGS[@]}"

echo "=========================================="
echo "Done : $(date)"
echo "=========================================="
