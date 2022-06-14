CUDA_VISIBLE_DEVICES=2 python gpt2_prefix_eval_oscar.py  --mapping_type transformer_encoder  --num_layers 8 --prefix_length 40 --prefix_length_clip 40 --checkpoint ../transformer_weights.pt
