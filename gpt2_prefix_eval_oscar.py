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


def train(data, model: ClipCaptionModel, out_path, tokenizer, args=None):
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

    images_root = "/home/dcor/datasets/COCO/val2014"
    if not os.path.isdir(images_root):
        images_root = "./data/coco/val2014"
    images_root = '/home/gamir/DER-Roei/davidn/CLIP_prefix_caption/data/coco/val2014/'
    embeddings = model.gpt.get_input_embeddings().weight.data
    embeddings = nnf.normalize(embeddings, 2, 1)
    skips = 0
    new_data = []
    results = []
    for ii, d in enumerate(data):
        #print(ii)
        #if ii-skips > 200:
         #   break

        img_id = d["image_id"]
        filename = f'{images_root}/COCO_val2014_{int(img_id):012d}.jpg'
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


def main():
    print('start....')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print('loaded tokenizer')
    sys.stdout.flush()
    # ron_dir = '/home/dcor/ronmokady/Oscar/checkpoint-29-66420'
    # amir_dir = './data/eval'
    # root_dir = ron_dir if os.path.isdir(ron_dir) else amir_dir
    # # with open(f'{root_dir}/pred.coco_caption.test.beam5.max20.odlabels_coco_format.json', 'r') as f:
    # #     data = json.load(f)
    # with open(f'david_params/captions_val2014.json', 'r') as f:
    #     data = json.load(f)['annotations']
    #
    images_root = "./data/coco/train2014"
    if not os.path.isdir(images_root):
        images_root = "./data/coco/val2014"
    # with open(f'data/coco/annotations/train_caption.json', 'r') as f:
    #     data = json.load(f)
    with open(f'/home/gamir/DER-Roei/davidn/CLIP_prefix_caption/data/coco/annotations/new_annotations/captions_val2014.json', 'r') as f:
        data = json.load(f)['annotations']

    clean_data_of_train_list = True
    if clean_data_of_train_list:
        train_list_img_ids = {}
        pt_train_list = '/home/gamir/DER-Roei/davidn/CLIP_prefix_caption/annotations/train_caption_of_real_training.json'
        with open(pt_train_list) as f:
            train_data = json.load(f)
        for d in train_data:
            train_list_img_ids[int(d['image_id'])] = 1
        i = 0
        for d in data:
            if int(d['image_id']) in train_list_img_ids:
                data.remove(d)
                i+=1
        print(f'\n{i} images removed from val data since they were in train data, the remaining data size is {len(data)}\n')

    root_dir = './'
    print('loaded data')
    print(type(data))
    print(len(data))
    print(data[0])
    data = data[1060:]
    for i in (10,):
        parser = argparse.ArgumentParser()
        #
        # parser.add_argument('--checkpoint', default=f'./checkpoints/oscar_split-007.pt')
        parser.add_argument('--checkpoint', default=f'./checkpoints/coco_prefix_t10_rn-006.pt')
        # parser.add_argument('--checkpoint2', default='./checkpoints/coco_train-012.pt')
        parser.add_argument('--only_prefix', dest='only_prefix', action='store_false')
        parser.add_argument('--beam', dest='beam', action='store_false')
        parser.add_argument('--is_rn', dest='is_rn', action='store_false')
        parser.add_argument('--prefix_length', type=int, default=i)
        parser.add_argument('--num_layers', type=int, default=8)
        parser.add_argument('--prefix_length_clip', type=int, default=10)
        parser.add_argument('--mapping_type', type=str, default='transformer_encoder',
                            help='mlp/transformer_encoder/transformer_decoder')
        args = parser.parse_args()

        name = args.checkpoint.split("/")[-1].split(".")[0] + ("_beam" if args.beam else "_max")
        out_path = f"{root_dir}/{name}.json"
        # if os.path.isfile(out_path):
        #     continue
        # model = ClipCaptionE2E()
        args.is_rn = 'rn' in args.checkpoint
        args.is_rn = True
        prefix_dim = [512, 640][args.is_rn]
        mapping_type = {'mlp': MappingType.MLP, 'transformer_encoder': MappingType.TransformerEncoder,
                        'transformer_decoder': MappingType.TransformerDecoder}[args.mapping_type]
        model = ClipCaptionModel(args.prefix_length, prefix_dim=prefix_dim, clip_length=args.prefix_length_clip,
                                  mapping_type=mapping_type, num_layers=args.num_layers)
        model.load_state_dict(torch.load(args.checkpoint, map_location=CUDA(0)))  # FIXME
        print(args.checkpoint)

        # model2 = ClipCaptionModel(10)
        # model2.load_state_dict(torch.load(args.checkpoint2))
        # print(args.checkpoint2)

        # for p1, p2 in zip(model.parameters(), model2.parameters()):
        #    print(p1.data - p2.data)

        # out_path = '/home/dcor/ronmokady/Oscar/checkpoint-29-66420/ours12_coco_format.json'

        train(data, model, out_path, tokenizer, args=args)
# def main():
#     for i in range(5,10):
#         parser = argparse.ArgumentParser()
#         parser.add_argument('--checkpoint', default=f'./checkpoints/coco_both_{i + 1}-005.pt')
#         #parser.add_argument('--checkpoint2', default='./checkpoints/coco_train-012.pt')
#         parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
#         parser.add_argument('--beam', dest='beam', action='store_false')
#         parser.add_argument('--prefix_length', type=int, default=i + 1)
#         parser.set_defaults(only_prefix=False)
#
#         args = parser.parse_args()
#
#         DEBUG = 0
#         batch_size = 10
#         num_epochs = 10
#
#
#         print("Loading JSON")
#         sys.stdout.flush()
#         ron_dir = '/home/dcor/ronmokady/Oscar/checkpoint-29-66420'
#         amir_dir = './data/eval'
#         root_dir = ron_dir if os.path.isdir(ron_dir) else amir_dir
#         with open(f'{root_dir}/pred.coco_caption.test.beam5.max20.odlabels_coco_format.json', 'r') as f:
#             data = json.load(f)
#
#         print(type(data))
#         print(len(data))
#         print(data[0])
#         # model = ClipCaptionE2E()
#         if args.only_prefix:
#             model = ClipCaptionPrefix(args.prefix_length)
#         else:
#             model = ClipCaptionModel(args.prefix_length)
#             print("Train both prefix and GPT")
#             sys.stdout.flush()
#
#         model.load_state_dict(torch.load(args.checkpoint, map_location=CPU)) #FIXME
#
#         print(args.checkpoint)
#
#         #model2 = ClipCaptionModel(10)
#         #model2.load_state_dict(torch.load(args.checkpoint2))
#         #print(args.checkpoint2)
#
#         #for p1, p2 in zip(model.parameters(), model2.parameters()):
#         #    print(p1.data - p2.data)
#
#         # out_path = '/home/dcor/ronmokady/Oscar/checkpoint-29-66420/ours12_coco_format.json'
#         name = args.checkpoint.split("/")[-1].split(".")[0] + ("_beam" if args.beam else "_max")
#         out_path = f"{root_dir}/{name}.json"
#         train(data, model, out_path, args=args)


if __name__ == '__main__':
    main()
    # images_root = "/home/dcor/datasets/COCO/val2014"
    # if not os.path.isdir(images_root):
    #     images_root = "./data/coco/val2014"
    # img_id = 525376
    # filename = f'{images_root}/COCO_val2014_{int(img_id):012d}.jpg'
    # image = Image.open(filename)
    # imshow(image)

# a display case full of different types of doughnuts.
# peanuts desserts elephbmÃÂÃÂÃ cooking nodd strikingly achieving\n'

#  display case filled with lots of different types of donuts.
#  glass bakery dough displays sandwiches2 boxes Prin ten
