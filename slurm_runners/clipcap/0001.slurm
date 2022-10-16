#!/bin/sh
#SBATCH --job-name=c0001
#SBATCH --output=0001.out
#SBATCH --error=0001.err
#SBATCH --partition=killable
#SBATCH --time=1400
#SBATCH --nodes=1
#SBATCH --nodelist=n-202
#SBATCH --ntasks=1
#SBATCH --gpus=1
cd /home/gamir/DER-Roei/davidn/CLIP_prefix_caption
python train.py --use_image_embedding_as_clipcap --data COCO --out /home/gamir/DER-Roei/davidn/CLIP_prefix_caption/slurm_runners/clipcap/results/0001/ --noise_variance 0.0001
python predictions_runner.py --checkpoint /home/gamir/DER-Roei/davidn/CLIP_prefix_caption/slurm_runners/clipcap/results/0001/coco_prefix-009.pt --dataset_mode 0