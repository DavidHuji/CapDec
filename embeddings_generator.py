import torch, random
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

# constants
# first line man, second woman, each column has the same form of gender. e.g. wife-husband, girl-boy etc.
gender_terms_map = [['boy', 'brother', 'dad', 'husband', 'man', 'groom', 'male',   'guy',  'men',  'males', 'boys',     'guys',   'dads', 'dude',    'policeman', 'policemen', 'boyfriend',      'father',  'son',     'fireman',      'he', 'actor', 'gentleman', 'mans', 'his', 'actors'],
                    ['girl', 'sister', 'mom', 'wife',  'woman', 'bride', 'female', 'lady', 'women', 'girls', 'ladies', 'females', 'moms', 'actress', 'nun',     'policewoman',    'girlfriend',  'mother',  'daughter',  'fire woman', 'she', 'actress', 'lady',  'women', 'her', 'actresses']]
gender_terms = gender_terms_map[0] + gender_terms_map[1]
gender_terms_set = set(gender_terms)
man_terms_set = set(gender_terms_map[0])
woman_terms_set = set(gender_terms_map[1])


def caption_has_gender_term(caption, gender_mode=0):  # gender_mode=0 for both, 1 for man only, 2 for woman only
    caption_words = caption.lower().split(' ')
    if gender_mode == 0:
        return len(set(caption_words) & gender_terms_set) > 0
    elif gender_mode==1:
        return len(set(caption_words) & man_terms_set) > 0
    elif gender_mode==2:
        return len(set(caption_words) & woman_terms_set) > 0


def change_gender_randomly(caption):
    caption_words = caption.lower().split(' ')

    for i in range(len(caption_words)):
        if caption_words[i] in gender_terms_set:
            form_index = gender_terms.index(caption_words[i]) % len(gender_terms_map[0])
            caption_words[i] = gender_terms_map[random.randint(0, 1)][form_index]
    new_caption = ' '.join(map(str, caption_words))
    print(f'Changed caption from {caption} to {new_caption}')
    return new_caption


def main(clip_model_type, clip_model_name, out_path, annotations_path, images_path, fix_gender_imbalance, data_mode):
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    all_text_embeddings = []
    long_caps = 0
    not_found = 0
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        if not add_text_embedding:
            if images_path != 'NoImgs':
                if data_mode == 0:
                    filename = f"./data/coco/train2014/COCO_train2014_{int(img_id):012d}.jpg"
                else:
                    filename = images_path + d['filename']
                if os.path.isfile(filename):
                    image = io.imread(filename)
                else:
                    not_found += 1
                    continue
                image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            if add_text_embedding:
                prefix = torch.tensor([])  # empty tensor
                caption = d["caption"]
                if fix_gender_imbalance:
                    if caption_has_gender_term(caption, gender_mode=(fix_gender_imbalance-1)):
                        caption = change_gender_randomly(caption)
                try:  # if caption is too long
                    caption_tokens = clip.tokenize(caption).to(device)
                except:
                    caption_tokens = clip.tokenize(caption[:100]).to(device)
                    long_caps += 1
                    print(f'Long captions: {long_caps} long caption: {caption}')
                caption_embedding = clip_model.encode_text(caption_tokens).cpu()
                # caption_embedding /= torch.norm(caption_embedding, keepdim=True) it is better to avoid normaliztion in this stage so it will be possible to normelise or not later
            else:
                prefix = clip_model.encode_image(image).cpu()

        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        if add_text_embedding:
            all_text_embeddings.append(caption_embedding)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions, 'clip_embedding_text_dave': torch.cat(all_text_embeddings, dim=0) if add_text_embedding else 0}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions, 'clip_embedding_text_dave': torch.cat(all_text_embeddings, dim=0) if add_text_embedding else 0}, f)

    print('Done')
    print(f'long_caps bigger then 76 amount was = {long_caps}')
    print("%0d embeddings saved " % len(all_embeddings))
    print(f'not found images = {not_found}')
    print(f'text embeddings = {add_text_embedding}')
    return 0


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="RN50x4", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--dataset_mode', type=float, default=0.0)  # 0 for COCO!!, 1 for flicker30, 2 humor style,3 romantic,4 factual of style,6 harrypotter, 7 for news.
    parser.add_argument('--fix_gender_imbalance_mode', type=int, default=0)  # 1 for both genders, 2 for man only, 3 for woman only
    args = parser.parse_args()
    clip_model_name = args.clip_model_type.replace('/', '_')
    if args.dataset_mode == 0:
        out_path = f"./data/coco/verified_split_COCO_train_set.pkl"
        if add_text_embedding:
            out_path = f"./data/coco/verified_split_COCO_train_set_with_text_not_norm.pkl"
            print(f'Text embeddings will be added to the dataset')
        annotations_path = f'/home/gamir/DER-Roei/davidn/myprivate_coco/annotations/train.json'
        images_path = '/home/gamir/DER-Roei/davidn/myprivate_coco/train2014/'
    elif args.dataset_mode == 0.5:
        out_path = f"./data/coco/COCO_val_set_single_cap_per_sample.pkl"
        if add_text_embedding:
            out_path = f"./data/coco/COCO_val_set_single_cap_per_sample_with_text_not_norm.pkl"
            print(f'Text embeddings will be added to the dataset')
        annotations_path = f'/home/gamir/DER-Roei/davidn/myprivate_coco/annotations/single_caption_per_sample_val.json'
        images_path = '/home/gamir/DER-Roei/davidn/myprivate_coco/val2014/'
    elif args.dataset_mode == 1:
        out_path = f"./data/flicker30_{clip_model_name}_train.pkl"
        if add_text_embedding:
            out_path = f"./data/flicker30_{clip_model_name}_train_with_text_embeddings_not_norm.pkl"
            print(f'Text embeddings will be added to the dataset')
        annotations_path = f"/home/gamir/DER-Roei/davidn/flicker30/dataset_flickr30k_correct_format.jsontrain"
        images_path = f"/home/gamir/DER-Roei/davidn/flicker30/flickr30k_images/"
    elif args.dataset_mode == 1.5:
        out_path = f"./data/flicker30_{clip_model_name}_validation.pkl"
        if add_text_embedding:
            out_path = f"./data/flicker30_{clip_model_name}_validation_with_text_embeddings.pkl"
            print(f'Text embeddings will be added to the dataset')
        annotations_path = f"/home/gamir/DER-Roei/davidn/flicker30/dataset_flickr30k_correct_format.jsonvalidation"
        images_path = f"/home/gamir/DER-Roei/davidn/flicker30/flickr30k_images/"
    elif args.dataset_mode == 2:
        out_path = f"./data/styleHumor_{clip_model_name}_train.pkl"
        if add_text_embedding:
            out_path = f"./data/styleHumor_{clip_model_name}_train_with_text_embeddings_not_norm.pkl"
            print(f'Text embeddings will be added to the dataset')
        annotations_path = f"/home/gamir/DER-Roei/davidn/flicker8kforStyle/postprocessed_style_data/humor_train.json"
        images_path = f'/home/gamir/DER-Roei/davidn/flicker8kforStyle/Images/'

    elif args.dataset_mode == 3:
        out_path = f"./data/styleRoman_{clip_model_name}_train.pkl"
        if add_text_embedding:
            out_path = f"./data/styleRoman_{clip_model_name}_train_with_text_embeddings_not_norm.pkl"
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
        out_path = f"./data/BALANCED_parsed_coco_snowboarding_split_train_MODEis{args.fix_gender_imbalance_mode}.pkl"
        annotations_path = f"/home/gamir/DER-Roei/davidn/CLIP_prefix_caption/coco_snowboarding_annnotations/my_coco_snowboarding_train.json"
        images_path = f'NoImgs'
    elif args.dataset_mode == 9:
        out_path = f"./data/shkspr_train.pkl"
        annotations_path = f"parssed_sheikspir_alllines_111k.json"
        images_path = f'NoImgs'
    print(f'out_path is {out_path} fix gender imbalance is {args.fix_gender_imbalance_mode}')
    exit(main(args.clip_model_type, clip_model_name, out_path, annotations_path, images_path, args.fix_gender_imbalance_mode, data_mode=args.dataset_mode))


if __name__ == '__main__':
    run_main()
