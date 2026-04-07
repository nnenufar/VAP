#!/bin/bash
#SBATCH --job-name=VAP_eval
#SBATCH --output=./slurm/out/eval.out
#SBATCH --error=./slurm/out/eval.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 8
#SBATCH --time=2-00:00:00
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s,a5000

source ~/miniconda3/bin/activate
conda activate vap
export WANDB_MODE=offline

VAD_EVENTS_PATH=data/eval/vad_events.csv

set -euo pipefail

# Extract VAD-based events
if [ -f $VAD_EVENTS_PATH ]; then
    echo "Events file exists"
else
    echo "Events file doesn't exist. Extracting..."
    python vap/data/dset_event.py \
    --audio_vad_csv data/splits/test.csv \
    --agg_csv /home/joao.lima/experiments/seamless_interaction/assets/interaction_aggregated.csv \
    --output $VAD_EVENTS_PATH \
    --pre_cond_time 1 \
    --post_cond_time 2 \
    --min_silence_time 0.1

    echo Sucessfully extracted VAD events.
fi

# Run evaluation
echo Running evaluation
python vap/eval_events.py \
    --checkpoint runs/checkpoints/epoch=14-step=105540.ckpt \
    --csv $VAD_EVENTS_PATH \
    --output data/eval/results/relCond \
    --batch_size 32 \
    --num_workers 8 \
    --prefetch_factor 2
    --plot

echo EVALUATION COMPLETE