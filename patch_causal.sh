#!/bin/bash
#SBATCH --job-name=reasoning_abstraction
#SBATCH --time=0-8:00:00 # D-HH:MM
#SBATCH --account=def-rgrosse
#SBATCH --mem=256G
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1

#salloc --account=def-zhijing --mem=512G --gpus=h100:2

# Load required modules
#module load python/3.11.5
#module load cuda/12.6
#module load scipy-stack/2023b
#module load arrow/21.0.0
module load python cuda scipy-stack arrow

source venv/bin/activate

pip install -e ../TransformerLens

python intervene_causal_mediation.py

