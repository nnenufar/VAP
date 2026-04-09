#!/bin/bash
#SBATCH --job-name=VAP_data
#SBATCH --output=./slurm/out/data.out
#SBATCH --error=./slurm/out/data.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 20
#SBATCH --time=2-00:00:00
#SBATCH --mem=15G

source ~/miniconda3/bin/activate
conda activate vap

set -euo pipefail

SI_DIR=/hadatasets/joao.lima/data/seamless_interaction/naturalistic/processed
PATHS_FULL_DIR=data/audio_vad_paths_full.csv
PATHS_SPLITS_DIR=data/splits

AGGREGATED_CSV=/home/joao.lima/experiments/seamless_interaction/assets/interaction_aggregated.csv

# Create full table with WAV and JSON paths
if [ -f $PATHS_FULL_DIR ]; then
    echo "Full paths file exists"
else
    python vap/data/create_audio_vad_csv.py \
        --audio_dir $SI_DIR/wavs \
        --vad_dir $SI_DIR/vad \
        --output $PATHS_FULL_DIR
    echo Successfully created full paths table

    # Create split-specific table from the full one
    python vap/data/create_splits.py \
        --csv $PATHS_FULL_DIR \
        --output_dir $PATHS_SPLITS_DIR \
        --train_size 0.8 \
        --val_size 0.15 \
        --test_size 0.05
    echo Successfully created split paths table
fi

# Create sliding window tables for each split. These will be the input to train.bash
python vap/data/create_sliding_window_dset.py \
    --audio_vad_csv $PATHS_SPLITS_DIR/train.csv \
    --agg_csv $AGGREGATED_CSV \
    --output $PATHS_SPLITS_DIR/train_WindowDset_fullPer.csv \
    --duration 20 \
    --overlap 5 \
    --horizon 2

python vap/data/create_sliding_window_dset.py \
    --audio_vad_csv $PATHS_SPLITS_DIR/test.csv \
    --agg_csv $AGGREGATED_CSV \
    --output $PATHS_SPLITS_DIR/test_WindowDset_fullPer.csv \
    --duration 20 \
    --overlap 5 \
    --horizon 2

python vap/data/create_sliding_window_dset.py \
    --audio_vad_csv $PATHS_SPLITS_DIR/val.csv \
    --agg_csv $AGGREGATED_CSV \
    --output $PATHS_SPLITS_DIR/val_WindowDset_fullPer.csv \
    --duration 20 \
    --overlap 5 \
    --horizon 2
echo Successfully created sliding window dataset tables for each split

# Sanity check
python vap/data/datamodule.py \
    --csv $PATHS_SPLITS_DIR/test_WindowDset_fullPer.csv \
    --batch_size 4 \
    --num_workers 2 \
    --prefetch_factor 2
echo Successfully passed datamodule test

echo DATA PROCESSING COMPLETE

