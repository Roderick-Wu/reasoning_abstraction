#!/bin/bash
#SBATCH --job-name=reasoning_abstraction
#SBATCH --time=0-2:00:00 # D-HH:MM
#SBATCH --account=def-zhijing
#SBATCH --mem=256G
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1

#salloc --account=def-zhijing --mem=128G --gpus=h100:1

# Load required modules
#module load python/3.11.5
#module load cuda/12.6
#module load scipy-stack/2023b
#module load arrow/21.0.0
module load python cuda scipy-stack arrow

source venv/bin/activate

pip install -e ../TransformerLens

#python linprob_cot.py
#python linprob_cot_all_layer.py
python linprob_cot_all_layer_time.py

