CUDA_VISIBLE_DEVICES=2 python predictions_runner.py  --mapping_type transformer_encoder  --num_layers 8 --prefix_length 40 --prefix_length_clip 40 --checkpoint ../transformer_weights.pt

for ablation distance:
add --dataset_mode 5 --text_autoencoder --ablation_dist --ablation_dist_review