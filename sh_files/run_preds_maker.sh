CUDA_VISIBLE_DEVICES=2 python embeddings_generator.py --checkpoint ../transformer_weights.pt --dataset_mode 0

for ablation distance: add --dataset_mode 5 --text_autoencoder --ablation_dist --ablation_dist_review