import sys
sys.path.append("/home/amir/projects/CLIP")
from transformers import GPT2Tokenizer
import os
from custom_types import *
import torch
from gpt2_prefix import ClipCaptionModel, MappingType
from PIL import Image
import matplotlib.pyplot as plt
import json
import clip   # installed from https://github.com/openai/CLIP
import argparse, pickle
from gpt2_prefix_eval import generate_beam, generate2, imshow, get_prefix_tokens
from torchvision import transforms
import os.path


def count_ready_parphrased_embeddings(embeddings_dict):
    ready = 0
    for img_id in embeddings_dict.keys():
        if embeddings_dict[img_id] is not None:
            if len(embeddings_dict[img_id]) == 5:
                ready += 1
    return ready


def get_precalculated_centers():
    with open('others/CLIP_embeddings_centers_info.pkl', 'rb') as f:
        return pickle.load(f)


def calc_distances_of_ready_embeddings(embeddings_dict, out_file='embeddings_distances.pkl'):
    # calculate the distance between the 5 prefixes
    distances, distances_l2, data_size = [], [], 0
    distances_clip, distances_l2_clip, max_distances_l1, distances_l2_from_center, max_distances_l1_from_center, maxoutof5 = [], [], [], [], [], []
    for img_id in embeddings_dict.keys():
        data_size += 1
        dist, dist_l2, combs, shape_pref = 0.0, 0.0, 0, 0
        dist_clip, dist_l2_clip, shape_pref_clip, max_distance_l1 = 0.0, 0.0, 0.0, 0.0
        distances_between_paraphrased_embeddings = []
        for i in range(len(embeddings_dict[img_id])):
            for j in range(i + 1, len(embeddings_dict[img_id])):
                dist += np.linalg.norm(embeddings_dict[img_id][i][0] -
                                       embeddings_dict[img_id][j][0], ord=1)
                dist_l2 += np.linalg.norm(embeddings_dict[img_id][i][0] -
                                          embeddings_dict[img_id][j][0], ord=2)
                shape_pref = embeddings_dict[img_id][i][0].shape[0]
                combs += 1
                dist_clip += np.linalg.norm(embeddings_dict[img_id][i][1] -
                                            embeddings_dict[img_id][j][1], ord=1)
                dist_l2_clip += np.linalg.norm(embeddings_dict[img_id][i][1] -
                                               embeddings_dict[img_id][j][1], ord=2)
                shape_pref_clip = embeddings_dict[img_id][i][1].shape[0]

                max_distance_l1 += np.abs(embeddings_dict[img_id][i][1] - embeddings_dict[img_id][j][1]).max()
                distances_between_paraphrased_embeddings.append(np.linalg.norm(embeddings_dict[img_id][i][1] -
                                               embeddings_dict[img_id][j][1], ord=2) / (shape_pref_clip ** 0.5))

        if combs == 5 * 4 / 2:
            # todo note that for l2 you should devide by sqrt(dim) rather than dim! for fix use *sqrt(sim)) later
            distances.append(dist / (shape_pref * combs))
            distances_l2.append(dist_l2 / (shape_pref * combs))
            distances_clip.append(dist_clip / (shape_pref_clip * combs))
            distances_l2_clip.append(dist_l2_clip / (shape_pref_clip * combs))
            max_distances_l1.append(max_distance_l1 / combs)
            maxoutof5.append(np.max(distances_between_paraphrased_embeddings))

        # calculate the distance from the center
        five_embeddings = np.array([s[1] for s in embeddings_dict[img_id]])
        center = five_embeddings.mean(axis=0)
        distances_l2_from_center.append(np.linalg.norm(five_embeddings - center, ord=2, axis=1).mean())
        max_distances_l1_from_center.append(np.abs(five_embeddings - center).max(axis=1).mean())
    print(
        f"\n\n\n Average noremlised L1 between 5 annotations of same image MAPPER: {np.array(distances).mean()}, STD: {np.array(distances).std()}")
    print(
        f"\n\n\n Average noremlised L2 between 5 annotations of same image MAPPER: {np.array(distances_l2).mean()}, STD: {np.array(distances_l2).std()}")
    print(
        f"\n\n\n Average noremlised L1 between 5 annotations of same image CLIP: {np.array(distances_clip).mean()}, STD: {np.array(distances_clip).std()}")
    print(
        f"\n\n\n Average noremlised L2 between 5 annotations of same image CLIP: {np.array(distances_l2_clip).mean()}, STD: {np.array(distances_l2_clip).std()}")
    print(
        f"\n\n\n Mean L2 between 5 annotations of same image CLIP to their center: {np.array(distances_l2_from_center).mean()}, STD: {np.array(distances_l2_from_center).std()}")
    print(
        f"\n\n\n Max (per-entry) L1 between 5 annotations of same image CLIP to their center: {np.array(max_distances_l1_from_center).mean()}, STD: {np.array(max_distances_l1_from_center).std()}")
    print(
        f"\n\n\n Max (per-entry) L1 between 5 annotations of same image CLIP: {np.array(max_distances_l1).mean()}, STD: {np.array(max_distances_l1).std()}")
    print(
        f"\n\n\n Taking max out of the 10 L2 between 5 annotations of same image CLIP: {np.array(maxoutof5).mean()}")
    if out_file is not None:
        import pickle
        with open(out_file, 'wb') as f:
            pickle.dump(({"distances_clip": distances_clip, "distances_l2_clip": distances_l2_clip, "max_distances_l1": max_distances_l1}), f)
        print(f"Saved distances to {out_file} and finished")
        exit(0)
    return distances, distances_l2, distances_clip, distances_l2_clip, data_size


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


class Timer:
    """
    measure inference time
    """
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.timings = []

    def __enter__(self):
        self.starter.record()
        return self

    def __exit__(self, *args):
        self.ender.record()
        torch.cuda.synchronize()
        interval = self.starter.elapsed_time(self.ender)
        self.timings.append(interval)
        self.sum += interval
        self.count += 1

    def __str__(self):
        mean_syn = self.sum / self.count
        std_syn = np.std(self.timings)
        return f"mean: {mean_syn:.2f} ms, std: {std_syn:.2f} ms"


def make_preds(data, model: ClipCaptionModel, out_path, tokenizer, dataset_mode, args=None):
    device = CUDA(0)
    model = model.to(device)
    model.eval()
    if args.is_rn:
        clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
        args.beam = True
    else:
        clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    # preprocess = clip_transform_full()
    #prefix_length = 10

    if args.add_modality_offset:
        modality_offset = get_precalculated_centers()['offset_to_add_in_inference'].to(device)

    if dataset_mode == 0 or dataset_mode == 7 or dataset_mode == 8:
        images_root = '/home/gamir/DER-Roei/davidn/CLIP_prefix_caption/data/coco/val2014'
        images_root = '../myprivate_coco/val2014'
    elif dataset_mode == 1:
        images_root = '/home/gamir/DER-Roei/davidn/flicker30/flickr30k_images'
    elif dataset_mode == 2 or dataset_mode == 3 or dataset_mode == 4:
        images_root = '/home/gamir/DER-Roei/davidn/flicker8kforStyle/Images'
    elif dataset_mode == 6:
        images_root = '/home/gamir/DER-Roei/davidn/CLIP_prefix_caption/data/coco/train2014'
        images_root = '../myprivate_coco/train2014'
    elif dataset_mode != 5:
        print("Wrong data mode")
        exit(1)

    if args.modality_bridger:
        from others.supervised_embedding_bridger import get_map_to_text_space_using_modality_bridger
        map_to_text_space_using_modality_bridger = get_map_to_text_space_using_modality_bridger()

    embeddings = model.gpt.get_input_embeddings().weight.data
    embeddings = nnf.normalize(embeddings, 2, 1)
    skips = 0
    new_data = []
    prefix_for_distance_ablation_metric = {}
    results = []
    ablation_image_dist_stat = {'counter': 0, 'L2': 0.0}
    timer = Timer()
    for ii, d in enumerate(data):
        img_id = d["image_id"]
        if dataset_mode == 0 or dataset_mode == 7 or dataset_mode == 8:
            filename = f'{images_root}/COCO_val2014_{int(img_id):012d}.jpg'
        elif dataset_mode == 6:
            filename = f'{images_root}/COCO_train2014_{int(img_id):012d}.jpg'
        elif dataset_mode == 1 or dataset_mode == 4 or dataset_mode == 2 or dataset_mode == 3:
            filename = d["filename"]
            filename = f'{images_root}/{filename}'
        elif dataset_mode == 5:
            filename = 'no need for filename, yay!!1'

        if not os.path.isfile(filename) and dataset_mode != 5:
            skips += 1
            print('skips=', skips, " filename=", filename)
            continue
        if dataset_mode != 5:
            image_raw = Image.open(filename).convert("RGB")
            image = preprocess(image_raw).unsqueeze(0).to(device)
        with torch.no_grad():
            timer.__enter__()
            if args.text_autoencoder or dataset_mode == 5:
                # in this case thew image is actually text input
                caption_tokens = clip.tokenize(d['caption']).to(device)
                prefix = clip_model.encode_text(caption_tokens).float()
            else:
                prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
            if not args.dont_normalize_prefix:
                prefix = prefix / prefix.norm(2, -1)
            if args.add_modality_offset:
                prefix = prefix + modality_offset
            if args.modality_bridger:
                prefix = map_to_text_space_using_modality_bridger(prefix)
                prefix / prefix.norm(2, -1)
            prefix_embed = model.clip_project(prefix).reshape(1, args.prefix_length, -1)
        if args.beam:
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
        timer.__exit__()
        results.append((img_id, d["caption"], generated_text_prefix.lower()))
        if args.ablation_dist:
            if d['image_id'] not in prefix_for_distance_ablation_metric:
                prefix_for_distance_ablation_metric[d['image_id']] = [(prefix_embed.cpu().numpy().reshape(-1), prefix.cpu().numpy().reshape(-1))]
            else:
                prefix_for_distance_ablation_metric[d['image_id']].append((prefix_embed.cpu().numpy().reshape(-1), prefix.cpu().numpy().reshape(-1)))
        if args.ablation_image_dist:
            with torch.no_grad():
                caption_tokens = clip.tokenize(d['caption']).to(device)
                txt_prefix = clip_model.encode_text(caption_tokens).float()
                txt_prefix /= txt_prefix.norm(2, -1)
            l2_dist_img_txt = np.linalg.norm(txt_prefix.cpu().numpy().reshape(-1) - prefix.cpu().numpy().reshape(-1), ord=2)
            ablation_image_dist_stat['counter'] += 1
            ablation_image_dist_stat['L2'] += l2_dist_img_txt

        if args.ablation_dist:  # this is for the review
            if count_ready_parphrased_embeddings(prefix_for_distance_ablation_metric) >= 900:
                calc_distances_of_ready_embeddings(prefix_for_distance_ablation_metric)

        if ii % 99 == 0:
            print(timer)
            for r in results:
                print(r)

            results.clear()
            print('\n\n', ii)
            with open(out_path, 'w') as outfile:
                json.dump(new_data, outfile)

            if args.ablation_dist:
                # calculate the distance between the 5 prefixes
                distances,distances_l2, data_size = [], [], 0
                distances_clip,distances_l2_clip = [],  []
                for img_id in prefix_for_distance_ablation_metric:
                    data_size += 1
                    dist, dist_l2, combs, shape_pref = 0.0, 0.0, 0, 0
                    dist_clip, dist_l2_clip, shape_pref_clip = 0.0, 0.0, 0
                    for i in range(len(prefix_for_distance_ablation_metric[img_id])):
                        for j in range(i + 1, len(prefix_for_distance_ablation_metric[img_id])):
                            dist += np.linalg.norm(prefix_for_distance_ablation_metric[img_id][i][0] -
                                                   prefix_for_distance_ablation_metric[img_id][j][0], ord=1)
                            dist_l2 += np.linalg.norm(prefix_for_distance_ablation_metric[img_id][i][0] -
                                                   prefix_for_distance_ablation_metric[img_id][j][0], ord=2)
                            shape_pref = prefix_for_distance_ablation_metric[img_id][i][0].shape[0]
                            combs += 1

                            dist_clip += np.linalg.norm(prefix_for_distance_ablation_metric[img_id][i][1] -
                                                   prefix_for_distance_ablation_metric[img_id][j][1], ord=1)
                            dist_l2_clip += np.linalg.norm(prefix_for_distance_ablation_metric[img_id][i][1] -
                                                   prefix_for_distance_ablation_metric[img_id][j][1], ord=2)
                            shape_pref_clip = prefix_for_distance_ablation_metric[img_id][i][1].shape[0]
                    if combs > 1:
                        distances.append(dist / (shape_pref * combs))
                        distances_l2.append(dist_l2 / (shape_pref * combs))
                        distances_clip.append(dist_clip / (shape_pref_clip * combs))
                        distances_l2_clip.append(dist_l2_clip / (shape_pref_clip * combs))
                print(f"\n\n\n Average noremlised L1 between 5 annotations of same image MAPPER: {np.array(distances).mean()}, STD: {np.array(distances).std()}")
                print(f"\n\n\n Average noremlised L2 between 5 annotations of same image MAPPER: {np.array(distances_l2).mean()}, STD: {np.array(distances_l2).std()}")
                print(f"\n\n\n Average noremlised L1 between 5 annotations of same image CLIP: {np.array(distances_clip).mean()}, STD: {np.array(distances_clip).std()}")
                print(f"\n\n\n Average noremlised L2 between 5 annotations of same image CLIP: {np.array(distances_l2_clip).mean()}, STD: {np.array(distances_l2_clip).std()}")
            if args.ablation_image_dist:
                print(f"\n\n\n L2 between images and texts embeddings: {ablation_image_dist_stat['L2'] / ablation_image_dist_stat['counter']}pr, dim size={prefix.shape}")
        if DEBUG and not args.ablation_dist and False:
            prefix_sent = get_prefix_tokens(prefix_embed, embeddings, tokenizer)
            imshow(image_raw, title=f'{generated_text_prefix}\n{prefix_sent}')

        d["caption"] = generated_text_prefix.lower()
        new_data.append({"caption": generated_text_prefix.lower(), "image_id": d["image_id"]})

    if args.ablation_dist:
        # calculate the distance between the 5 prefixes
        distances, distances_l2, data_size = [], [], 0
        distances_clip, distances_l2_clip = [], []
        for img_id in prefix_for_distance_ablation_metric:
            data_size += 1
            dist, dist_l2, combs, shape_pref = 0.0, 0.0, 0, 0
            dist_clip, dist_l2_clip, shape_pref_clip = 0.0, 0.0, 0,
            for i in range(len(prefix_for_distance_ablation_metric[img_id])):
                for j in range(i + 1, len(prefix_for_distance_ablation_metric[img_id])):
                    dist += np.linalg.norm(prefix_for_distance_ablation_metric[img_id][i][0] -
                                           prefix_for_distance_ablation_metric[img_id][j][0], ord=1)
                    dist_l2 += np.linalg.norm(prefix_for_distance_ablation_metric[img_id][i][0] -
                                              prefix_for_distance_ablation_metric[img_id][j][0], ord=2)
                    shape_pref = prefix_for_distance_ablation_metric[img_id][i][0].shape[0]
                    combs += 1

                    dist_clip += np.linalg.norm(prefix_for_distance_ablation_metric[img_id][i][1] -
                                                prefix_for_distance_ablation_metric[img_id][j][1], ord=1)
                    dist_l2_clip += np.linalg.norm(prefix_for_distance_ablation_metric[img_id][i][1] -
                                                   prefix_for_distance_ablation_metric[img_id][j][1], ord=2)
                    shape_pref_clip = prefix_for_distance_ablation_metric[img_id][i][1].shape[0]
            if combs > 1:
                distances.append(dist / (shape_pref * combs))
                distances_l2.append(dist_l2 / (shape_pref * combs))
                distances_clip.append(dist_clip / (shape_pref_clip * combs))
                distances_l2_clip.append(dist_l2_clip / (shape_pref_clip * combs))
        print(
            f"\n\n\n Average noremlised L1 between 5 annotations of same image MAPPER: {np.array(distances).mean()}, STD: {np.array(distances).std()}")
        print(
            f"\n\n\n Average noremlised L2 between 5 annotations of same image MAPPER: {np.array(distances_l2).mean()}, STD: {np.array(distances_l2).std()}")
        print(
            f"\n\n\n Average noremlised L1 between 5 annotations of same image CLIP: {np.array(distances_clip).mean()}, STD: {np.array(distances_clip).std()}")
        print(
            f"\n\n\n Average noremlised L2 between 5 annotations of same image CLIP: {np.array(distances_l2_clip).mean()}, STD: {np.array(distances_l2_clip).std()}")
    if args.ablation_image_dist:
        print(
            f"\n\n\nFinal L2 between images and texts embeddings: {ablation_image_dist_stat['L2'] / ablation_image_dist_stat['counter']}, dim size={prefix.shape}")

    return 0


def load_data(dataset_mode):
    if dataset_mode == 0:
        with open(
                f'/home/gamir/DER-Roei/davidn/myprivate_coco/annotations/single_caption_per_sample_val.json',
                'r') as f:
            data = json.load(f)
    elif dataset_mode == 1:
        with open(
                f'/home/gamir/DER-Roei/davidn/flicker30/dataset_flickr30k_correct_format.jsonvalidation',
                'r') as f:
            data = json.load(f)
    elif dataset_mode == 2:
        with open(
                f'/home/gamir/DER-Roei/davidn/flicker8kforStyle/postprocessed_style_data/humor_test.json',
                'r') as f:
            data = json.load(f)
    elif dataset_mode == 3:
        with open(
                f'/home/gamir/DER-Roei/davidn/flicker8kforStyle/postprocessed_style_data/roman_test.json',
                'r') as f:
            data = json.load(f)
    elif dataset_mode == 4:
        with open(
                f'/home/gamir/DER-Roei/davidn/flicker8kforStyle/postprocessed_style_data/factual_test.json',
                'r') as f:
            data = json.load(f)
    elif dataset_mode == 5:
        with open(
                f'/home/gamir/DER-Roei/davidn/myprivate_coco/annotations/val.json',
                'r') as f:
            data = json.load(f)
    elif dataset_mode == 6:
        with open(
                f'/home/gamir/DER-Roei/davidn/myprivate_coco/annotations/train.json',
                'r') as f:
            data = json.load(f)
    elif dataset_mode == 7:
        with open(f'/home/gamir/DER-Roei/davidn/CLIP_prefix_caption/coco_snowboarding_annnotations/my_coco_snowboarding_test.json', 'r') as f:
            data = json.load(f)
    elif dataset_mode == 8:
        with open(f'/home/gamir/DER-Roei/davidn/CLIP_prefix_caption/combinedNwes_on_cocoVal.json', 'r') as f:
            data = json.load(f)
    else:
        print("Wrong dataset mode")
        exit(3)

    clean_data_of_train_list = False and (dataset_mode == 0)  # only for coco
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=f'./checkpoints/coco_prefix_t10_rn-006.pt')
    parser.add_argument('--out', default='')
    parser.add_argument('--dataset_mode', type=int, default=0)  # 0 for coco val, 1 for flicker30, 2 humor style,3 romantic,4 factual of style, 5 coco val text only, 6 coco train, 7 coco val for womanSnowboard_for_creating_capdec_preds
    parser.add_argument('--modality_bridger', dest='modality_bridger', action='store_true', default=False)
    parser.add_argument('--beam', dest='beam', action='store_true', default=True)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true', default=True)
    parser.add_argument('--dont_normalize_prefix', dest='dont_normalize_prefix', action='store_true', default=False)
    parser.add_argument('--text_autoencoder', dest='text_autoencoder', action='store_true', default=False)
    parser.add_argument('--add_modality_offset', dest='add_modality_offset', action='store_true', default=False)
    parser.add_argument('--ablation_dist', dest='ablation_dist', action='store_true', default=False)  # need to use dataset_mode=5 to use only text
    parser.add_argument('--ablation_image_dist', dest='ablation_image_dist', action='store_true', default=False)
    parser.add_argument('--prefix_length', type=int, default=40)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--prefix_length_clip', type=int, default=40)
    parser.add_argument('--mapping_type', type=str, default='transformer_encoder',
                        help='mlp/transformer_encoder/transformer_decoder')
    args = parser.parse_args()
    print(f'beam search = {args.beam}')
    if args.text_autoencoder:
        args.dataset_mode = 5
    data = load_data(dataset_mode=args.dataset_mode)
    name = args.checkpoint.split("/")[-1].split(".")[0] + ('add_modality_offset' if args.add_modality_offset else '')
    checkpoint_dir = '/'.join(args.checkpoint.split("/")[:-1])
    out_path = f"{checkpoint_dir}/{name}.json" if (args.out == '') else args.out
    print(f'out_path = {out_path}, dataset_mode = {args.dataset_mode}')

    out_dir = '/'.join(out_path.split('/')[:-1])
    with open(f'{out_dir}/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        print(f'args saved to file {out_dir}/pred_commandline_args.txt')

    prefix_dim = [512, 640][args.is_rn]
    mapping_type = {'mlp': MappingType.MLP, 'transformer_encoder': MappingType.TransformerEncoder,
                    'transformer_decoder': MappingType.TransformerDecoder}[args.mapping_type]
    model = ClipCaptionModel(args.prefix_length, prefix_dim=prefix_dim, clip_length=args.prefix_length_clip,
                              mapping_type=mapping_type, num_layers=args.num_layers)
    model.load_state_dict(torch.load(args.checkpoint, map_location=CUDA(0)))  # FIXME
    print(args.checkpoint)
    print(f'modality_offset={args.add_modality_offset}')

    make_preds(data, model, out_path, tokenizer, args.dataset_mode, args=args)


if __name__ == '__main__':
    main()
