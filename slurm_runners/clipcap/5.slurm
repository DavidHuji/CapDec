#!/bin/sh
#SBATCH --job-name=cap5
#SBATCH --output=5.out
#SBATCH --error=5.err
#SBATCH --partition=killable
#SBATCH --time=1400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --nodelist=n-204
cd /home/gamir/DER-Roei/davidn/CLIP_prefix_caption
python train.py --use_image_embedding_as_clipcap --data COCO --out /home/gamir/DER-Roei/davidn/CLIP_prefix_caption/slurm_runners/clipcap/results/5/ --noise_variance 0.5
python predictions_runner.py --checkpoint /home/gamir/DER-Roei/davidn/CLIP_prefix_caption/slurm_runners/clipcap/results/5/coco_prefix-009.pt --dataset_mode 5