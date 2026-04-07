#!/bin/bash
#SBATCH --job-name=VAP_train
#SBATCH --output=./slurm/out/train.out
#SBATCH --error=./slurm/out/train.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 8
#SBATCH --time=2-00:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=h100

source ~/miniconda3/bin/activate
conda activate vap

export WANDB_MODE=offline

python vap/main.py \
  --config-name=relComb_cat_train_config \
  datamodule.train_path=data/splits/train_WindowDset.csv \
  datamodule.val_path=data/splits/val_WindowDset.csv \
  datamodule.test_path=data/splits/test_WindowDset.csv \
  datamodule.batch_size=64 \
  datamodule.num_workers=8