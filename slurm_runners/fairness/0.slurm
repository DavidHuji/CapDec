#!/bin/sh
#SBATCH --job-name=fairness0
#SBATCH --output=0.out
#SBATCH --error=0.err
#SBATCH --partition=killable
#SBATCH --time=1400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
cd /home/gamir/DER-Roei/davidn/CLIP_prefix_caption
python train.py --data ./data/parsed_coco_snowboarding_split_train.pkl --out /home/gamir/DER-Roei/davidn/CLIP_prefix_caption/slurm_runners/fairness/results/0/ --noise_variance 0.025
python predictions_runner.py --checkpoint /home/gamir/DER-Roei/davidn/CLIP_prefix_caption/slurm_runners/fairness/results/0/coco_prefix-009.pt --dataset_mode 7
