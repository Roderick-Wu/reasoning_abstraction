#!/bin/bash
# Submit Experiment C for all 8 hidden-variable experiments.
# Run from the reasoning_abstraction/ directory:
#   bash submit_experiment_c_all.sh
#
# To run a subset:
#   bash submit_experiment_c_all.sh velocity current radius

EXPERIMENTS=("velocity" "current" "radius" "side_length" "wavelength" "cross_section" "displacement" "market_cap")

# If arguments provided, use those instead
if [[ $# -gt 0 ]]; then
    EXPERIMENTS=("$@")
fi

mkdir -p slurm_logs

for exp in "${EXPERIMENTS[@]}"; do
    JOB_ID=$(sbatch --job-name="exp_c_${exp}" experiment_c.sh "${exp}" | awk '{print $NF}')
    echo "Submitted ${exp}  ->  job ${JOB_ID}"
done

echo ""
echo "Check status with:  squeue -u $USER"
echo "Results will appear in: intervention_token_results/experiment_c_<exp>_results.json"
