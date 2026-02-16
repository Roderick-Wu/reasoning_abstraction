#!/bin/bash
#SBATCH --job-name=reasoning_abstraction_generate_data
#SBATCH --time=0-5:00:00 # D-HH:MM
#SBATCH --account=def-zhijing
#SBATCH --mem=512G
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=1

#salloc --account=def-zhijing --mem=128G --gpus=h100:2

# Load required modules
#module load python/3.11.5
#module load cuda/12.6
#module load scipy-stack/2023b
#module load arrow/21.0.0
module load python cuda scipy-stack arrow

source venv/bin/activate

pip install -e ../TransformerLens

#python generate_traces.py --experiment current --n_prompts 200 --max_new_tokens 256 --model_path /home/wuroderi/projects/def-zhijing/wuroderi/models/QwQ-32B-Preview
python generate_traces.py --experiment $1 --n_prompts 200 --max_new_tokens 256 --model_path /home/wuroderi/projects/def-zhijing/wuroderi/models/Qwen2.5-32B
