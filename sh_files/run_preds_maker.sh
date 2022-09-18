CUDA_VISIBLE_DEVICES=3 python predictions_runner.py --checkpoint fairness_mode2_training/best_weights.pt --dataset_mode 7 --out fairness_mode2_training/preds_fairness_mode2.json
for ablation distance: add --dataset_mode 5 --text_autoencoder --ablation_dist --ablation_dist_review


CUDA_VISIBLE_DEVICES=1 python predictions_runner.py  --dataset_mode 0 --checkpoint ./verification_train/000001/best.pt --out ./verification_train/000001/preds.json
