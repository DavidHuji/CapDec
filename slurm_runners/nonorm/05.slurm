#!/bin/sh
#SBATCH --job-name=coco05
#SBATCH --output=05.out
#SBATCH --error=05.err
#SBATCH --partition=killable
#SBATCH --time=1400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
cd /home/gamir/DER-Roei/davidn/CLIP_prefix_caption
python train.py --dont_norm --data COCO --out /home/gamir/DER-Roei/davidn/CLIP_prefix_caption/slurm_runners/results/05/ --noise_variance 0.0025
python predictions_runner.py --checkpoint /home/gamir/DER-Roei/davidn/CLIP_prefix_caption/slurm_runners/results/05/coco_prefix-009.pt --dataset_mode 0