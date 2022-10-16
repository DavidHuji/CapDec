#!/bin/sh
#SBATCH --job-name=autoencoder
#SBATCH --output=autoencoder.out
#SBATCH --error=autoencoder.err
#SBATCH --partition=killable
#SBATCH --time=1400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --nodelist=n-204|n-203|n-202
cd /home/gamir/DER-Roei/davidn/CLIP_prefix_caption
python predictions_runner.py --checkpoint final_repro/coco/capdec/0/coco_prefix-009.pt --text_autoencoder
python predictions_runner.py --checkpoint final_repro/coco/capdec/001/coco_prefix-009.pt --text_autoencoder
python predictions_runner.py --checkpoint final_repro/coco/capdec/01/coco_prefix-009.pt --text_autoencoder
python predictions_runner.py --checkpoint final_repro/coco/capdec/1/coco_prefix-009.pt --text_autoencoder
python predictions_runner.py --checkpoint final_repro/coco/capdec/25_0/coco_prefix-009.pt --text_autoencoder
python predictions_runner.py --checkpoint final_repro/coco/capdec/0001/coco_prefix-009.pt --text_autoencoder
python predictions_runner.py --checkpoint final_repro/coco/capdec/005/coco_prefix-009.pt --text_autoencoder
python predictions_runner.py --checkpoint final_repro/coco/capdec/015/coco_prefix-009.pt --text_autoencoder
python predictions_runner.py --checkpoint final_repro/coco/capdec/025/coco_prefix-009.pt --text_autoencoder
python predictions_runner.py --checkpoint final_repro/coco/capdec/05/coco_prefix-009.pt --text_autoencoder
python predictions_runner.py --checkpoint final_repro/coco/capdec/5/coco_prefix-009.pt --text_autoencoder
