#!/bin/sh
#SBATCH --job-name=coco15
#SBATCH --output=15.out
#SBATCH --error=15.err
#SBATCH --partition=killable
#SBATCH --time=1400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
cd /home/gamir/DER-Roei/davidn/CLIP_prefix_caption
python train.py --dont_norm --data COCO --out /home/gamir/DER-Roei/davidn/CLIP_prefix_caption/slurm_runners/results/15/ --noise_variance 0.0225
python predictions_runner.py --checkpoint /home/gamir/DER-Roei/davidn/CLIP_prefix_caption/slurm_runners/results/15/coco_prefix-009.pt --dataset_mode 0