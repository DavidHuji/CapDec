# import json
#
#
# def fix_flicker_format(json_file, new_path):
#     with open(json_file) as f:
#         data = json.load(f)['images']
#     new_data_train = []
#     new_data_validation = []
#     for d in data:
#         for sentence in d['sentences']:
#             if d['split'] == 'train':
#                 new_data_train.append({"image_id": d["imgid"], "caption": sentence["raw"], "id": sentence["sentid"]})
#             else:
#                 new_data_validation.append({"image_id": d["imgid"], "caption": sentence["raw"], "id": sentence["sentid"]})
#
#     with open(new_path + 'train', 'w') as f:
#         json.dump(new_data_train, f)
#     with open(new_path + 'validation', 'w') as f:
#         json.dump(new_data_validation, f)
#
#     print(f'New format data saved at {new_path} sizes:{len(new_data_train)}+{len(new_data_validation)} {len(new_data_train) + len(new_data_validation)} Old size is {len(data)*5}')
#
#
# if __name__ == '__main__':
#     pt = 'annotations/dataset_flickr30k.json'
#     new_name = 'annotations/dataset_flickr30k_correct_format.json'
#     fix_flicker_format(pt, new_name)

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
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
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
                caption_tokens = clip.tokenize(caption).to(device)
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
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    clip_model_name = args.clip_model_type.replace('/', '_')
    out_path = f"./data/flicker30_{clip_model_name}_train.pkl"
    if add_text_embedding:
        out_path = f"./data/flicker30_{clip_model_name}_train_with_text_embeddings.pkl"
        print(f'Text embeddings will be added to the dataset')

    annotations_path = f"/home/gamir/DER-Roei/davidn/flicker30/dataset_flickr30k_correct_format.jsontrain"
    images_path = f"/home/gamir/DER-Roei/davidn/flicker30/flickr30k_images/"

    exit(main(args.clip_model_type, clip_model_name, out_path, annotations_path, images_path))


if __name__ == '__main__':
    run_main()
