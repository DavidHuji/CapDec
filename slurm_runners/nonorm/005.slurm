#!/bin/sh
#SBATCH --job-name=coco005
#SBATCH --output=005.out
#SBATCH --error=005.err
#SBATCH --partition=killable
#SBATCH --time=1400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
cd /home/gamir/DER-Roei/davidn/CLIP_prefix_caption
python train.py --dont_norm --data COCO --out /home/gamir/DER-Roei/davidn/CLIP_prefix_caption/slurm_runners/results/005/ --noise_variance 0.000025
python predictions_runner.py --checkpoint /home/gamir/DER-Roei/davidn/CLIP_prefix_caption/slurm_runners/results/005/coco_prefix-009.pt --dataset_mode 0