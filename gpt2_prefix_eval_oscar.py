import sys
sys.path.append("/home/amir/projects/CLIP")
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import os
from custom_types import *
from tqdm import tqdm, trange
import torch
from gpt2_prefix import ClipCocoDataset, ClipCaptionModel, ClipCaptionPrefix, MappingType
from PIL import Image
import matplotlib.pyplot as plt
import json
import clip   # installed from https://github.com/openai/CLIP
import argparse
from gpt2_prefix_eval import generate_beam, generate2, imshow, get_prefix_tokens
from gpt2_prefix_e2e import ClipCaptionE2E
from torchvision import transforms
# from oscar_eval_amir_ig_



def image_to_display(img) -> ARRAY:
    if type(img) is str:
        img = Image.open(str(img))
    if type(img) is not V:
        img = V(img)
    return img


def imshow(img, title: Optional[str] = None):
    img = image_to_display(img)
    plt.imshow(img)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close('all')


def clip_transform_full(n_px=224):
    return transforms.Compose([
        transforms.Resize((n_px, n_px), interpolation=Image.BICUBIC),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    
import os.path


def make_preds(data, model: ClipCaptionModel, out_path, tokenizer, data_mode, args=None):
    device = CUDA(0)
    model = model.to(device)
    model.eval()
    if args.is_rn:
        clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
        normalize = True
        args.beam = True
    else:
        clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        normalize = False
    # preprocess = clip_transform_full()
    #prefix_length = 10

    if data_mode == 0:
        images_root = '/home/gamir/DER-Roei/davidn/CLIP_prefix_caption/data/coco/val2014/'
    else:
        images_root = '/home/gamir/DER-Roei/davidn/flicker30/flickr30k_images'
    embeddings = model.gpt.get_input_embeddings().weight.data
    embeddings = nnf.normalize(embeddings, 2, 1)
    skips = 0
    new_data = []
    results = []
    for ii, d in enumerate(data):

        img_id = d["image_id"]
        if data_mode == 0:
            filename = f'{images_root}/COCO_val2014_{int(img_id):012d}.jpg'
        elif data_mode == 1:
            filename = d["filename"]
            filename = f'{images_root}/{filename}'
        #print(filename)
        if not os.path.isfile(filename):
            skips += 1
            print('skips=', skips, " filename=", filename)
            continue
        image_raw = Image.open(filename).convert("RGB")
        image = preprocess(image_raw).unsqueeze(0).to(device)
        with torch.no_grad():
            if type(model) is ClipCaptionE2E:
                prefix_embed = model.forward_image(image)
            else:
                prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
                if normalize:
                    prefix = prefix / prefix.norm(2, -1)
                prefix_embed = model.clip_project(prefix).reshape(1, args.prefix_length, -1)
        if args.beam:
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

        results.append((img_id, d["caption"], generated_text_prefix.lower()))
        if ii % 1999 == 0:
            print('\n\n', ii, results)
            results.clear()
            print('\n\n', ii)
            with open(out_path, 'w') as outfile:
                json.dump(new_data, outfile)
        if DEBUG:
            prefix_sent = get_prefix_tokens(prefix_embed, embeddings, tokenizer)
            imshow(image_raw, title=f'{generated_text_prefix}\n{prefix_sent}')

        d["caption"] = generated_text_prefix.lower()
        new_data.append({"caption": generated_text_prefix.lower(), "image_id": d["image_id"]})


    #sys.exit()
    with open(out_path, 'w') as outfile:
        json.dump(new_data, outfile)
    print("JSON is dumped", " skipped=", skips)

    return 0


def load_data(dataset_mode):
    if dataset_mode == 0:
        with open(
                f'/home/gamir/DER-Roei/davidn/CLIP_prefix_caption/data/coco/annotations/new_annotations/captions_val2014.json',
                'r') as f:
            data = json.load(f)['annotations']
    elif dataset_mode == 1:
        with open(
                f'/home/gamir/DER-Roei/davidn/flicker30/dataset_flickr30k_correct_format.jsonvalidation',
                'r') as f:
            data = json.load(f)

    clean_data_of_train_list = True and (dataset_mode == 0)  # only for coco
    if clean_data_of_train_list:
        train_list_img_ids = {}
        pt_train_list = '/home/gamir/DER-Roei/davidn/CLIP_prefix_caption/data/coco/annotations/train_caption.json'
        with open(pt_train_list) as f:
            train_data = json.load(f)
        for d in train_data:
            train_list_img_ids[int(d['image_id'])] = 1
        i = 0
        for d in data:
            if int(d['image_id']) in train_list_img_ids:
                data.remove(d)
                i += 1
        print(f'\n{i} images removed from val data since they were in train data, the remaining data size is {len(data)}\n')


    print('loaded data')
    print(type(data))
    print(len(data))
    print("sample example: ", data[0])
    return data


def main():
    print('start....')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print('loaded tokenizer')
    sys.stdout.flush()

    images_root = "./data/coco/train2014"
    if not os.path.isdir(images_root):
        images_root = "./data/coco/val2014"
    #
    root_dir = './'
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=f'./checkpoints/coco_prefix_t10_rn-006.pt')
    # parser.add_argument('--checkpoint2', default='./checkpoints/coco_train-012.pt')
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_false')
    parser.add_argument('--beam', dest='beam', action='store_false')
    parser.add_argument('--is_rn', dest='is_rn', action='store_false')
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--dataset_mode', type=int, default=0)  # 0 for coco, 1 for flicker30
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--mapping_type', type=str, default='transformer_encoder',
                        help='mlp/transformer_encoder/transformer_decoder')
    args = parser.parse_args()
    data = load_data(dataset_mode=args.dataset_mode)

    name = args.checkpoint.split("/")[-1].split(".")[0] + ("_beam" if args.beam else "_max")
    out_path = f"{root_dir}/{name}.json"

    args.is_rn = 'rn' in args.checkpoint
    args.is_rn = True
    prefix_dim = [512, 640][args.is_rn]
    mapping_type = {'mlp': MappingType.MLP, 'transformer_encoder': MappingType.TransformerEncoder,
                    'transformer_decoder': MappingType.TransformerDecoder}[args.mapping_type]
    model = ClipCaptionModel(args.prefix_length, prefix_dim=prefix_dim, clip_length=args.prefix_length_clip,
                              mapping_type=mapping_type, num_layers=args.num_layers)
    model.load_state_dict(torch.load(args.checkpoint, map_location=CUDA(0)))  # FIXME
    print(args.checkpoint)
    make_preds(data, model, out_path, tokenizer, args.dataset_mode, args=args)


if __name__ == '__main__':
    main()
