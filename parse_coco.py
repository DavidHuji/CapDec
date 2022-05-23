import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse

# dave conf
# ~~~~~~~~~
add_text_embedding = True


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/oscar_split_{clip_model_name}_train.pkl"
    if add_text_embedding:
        out_path = f"./data/coco/oscar_split_{clip_model_name}_train_with_text_embeddings.pkl"
        print(f'Text embeddings will be added to the dataset')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('./data/coco/annotations/train_caption.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    all_text_embeddings = []
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        filename = f"./data/coco/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"./data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
