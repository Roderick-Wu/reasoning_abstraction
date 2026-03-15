#!/bin/bash
#SBATCH --job-name=reasoning_abstraction_generate_data
#SBATCH --time=0-4:00:00 # D-HH:MM
#SBATCH --account=def-rgrosse
#SBATCH --mem=128G
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1

#salloc --account=def-zhijing --mem=256G --gpus=h100:2

# Load required modules
#module load python/3.11.5
#module load cuda/12.6
#module load scipy-stack/2023b
#module load arrow/21.0.0
module load python cuda scipy-stack arrow

source venv/bin/activate

pip install -e ../TransformerLens

EXPERIMENT=${1:-velocity}

python cot_early_termination.py --experiment "${EXPERIMENT}" --n_variations 100