import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse, math

# dave conf
# ~~~~~~~~~
add_text_embedding = True
device = torch.device('cuda:0')


def main(clip_model_type, clip_model_name, out_path, annotations_path, images_path):

    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    all_text_embeddings = []
    long_caps = 0 
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        if images_path != 'NoImgs':
            filename = images_path + d['filename']
            image = io.imread(filename)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            if not add_text_embedding:
                prefix = clip_model.encode_image(image).cpu()
            else:
                prefix = torch.tensor([])
            if add_text_embedding:
                caption = d["caption"]
                try:  # if caption is too long
                    caption_tokens = clip.tokenize(caption).to(device)
                except:
                    caption_tokens = clip.tokenize(caption[100]).to(device)
                    long_caps += 1
                    print(f'Long captions: {long_caps} long caption: {caption}')
                caption_embedding = clip_model.encode_text(caption_tokens).cpu()
                caption_embedding /= torch.norm(caption_embedding, keepdim=True)
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        if add_text_embedding:
            all_text_embeddings.append(caption_embedding)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions, 'clip_embedding_text_dave': torch.cat(all_text_embeddings, dim=0)}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions, 'clip_embedding_text_dave': torch.cat(all_text_embeddings, dim=0)}, f)

    print('Done')
    print(f'long_caps bigger then 76 amount was = {long_caps}')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="RN50x4", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--dataset_mode', type=int, default=1)  # 0 for NOTHING!!, 1 for flicker30, 2 humor style,3 romantic,4 factual of style,6 harrypotter, 7 for news.
    args = parser.parse_args()
    clip_model_name = args.clip_model_type.replace('/', '_')
    if args.dataset_mode == 1:
        out_path = f"./data/flicker30_{clip_model_name}_train.pkl"
        if add_text_embedding:
            out_path = f"./data/flicker30_{clip_model_name}_train_with_text_embeddings.pkl"
            print(f'Text embeddings will be added to the dataset')
        annotations_path = f"/home/gamir/DER-Roei/davidn/flicker30/dataset_flickr30k_correct_format.jsontrain"
        images_path = f"/home/gamir/DER-Roei/davidn/flicker30/flickr30k_images/"
    elif args.dataset_mode == 2:
        out_path = f"./data/styleHumor_{clip_model_name}_train.pkl"
        if add_text_embedding:
            out_path = f"./data/styleHumor_{clip_model_name}_train_with_text_embeddings.pkl"
            print(f'Text embeddings will be added to the dataset')
        annotations_path = f"/home/gamir/DER-Roei/davidn/flicker8kforStyle/postprocessed_style_data/humor_train.json"
        images_path = f'/home/gamir/DER-Roei/davidn/flicker8kforStyle/Images/'

    elif args.dataset_mode == 3:
        out_path = f"./data/styleRoman_{clip_model_name}_train.pkl"
        if add_text_embedding:
            out_path = f"./data/styleRoman_{clip_model_name}_train_with_text_embeddings.pkl"
            print(f'Text embeddings will be added to the dataset')
        annotations_path = f"/home/gamir/DER-Roei/davidn/flicker8kforStyle/postprocessed_style_data/roman_train.json"
        images_path = f'/home/gamir/DER-Roei/davidn/flicker8kforStyle/Images/'
    elif args.dataset_mode == 4:
        out_path = f"./data/styleFactual_{clip_model_name}_train.pkl"
        if add_text_embedding:
            out_path = f"./data/styleFactual_{clip_model_name}_train_with_text_embeddings.pkl"
            print(f'Text embeddings will be added to the dataset')
        annotations_path = f"/home/gamir/DER-Roei/davidn/flicker8kforStyle/postprocessed_style_data/factual_train.json"
        images_path = f'/home/gamir/DER-Roei/davidn/flicker8kforStyle/Images/'
    elif args.dataset_mode == 6:
        out_path = f"./data/hp_train.pkl"
        annotations_path = f"parssed_harryPotterBooks.json"
        images_path = f'NoImgs'
    elif args.dataset_mode == 7:
        out_path = f"./data/parsed_news_train.pkl"
        annotations_path = f"parssed_news_data.json"
        images_path = f'NoImgs'
    elif args.dataset_mode == 8:
        out_path = f"./data/parsed_coco_snowboarding_split_train.pkl"
        annotations_path = f"/home/gamir/DER-Roei/davidn/CLIP_prefix_caption/coco_snowboarding_annnotations/my_coco_snowboarding_train.json"
        images_path = f'NoImgs'

    exit(main(args.clip_model_type, clip_model_name, out_path, annotations_path, images_path))


if __name__ == '__main__':
    run_main()
