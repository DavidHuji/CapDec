from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import os
from custom_types import *
from tqdm import tqdm, trange
import torch
from gpt2_prefix import ClipCocoDataset, ClipCaptionModel
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt


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


class ClipCocoDatasetWithImages(ClipCocoDataset):

    def __getitem__(self, item):
        tokens, mask, prefix, caption = super(ClipCocoDatasetWithImages, self).__getitem__(item)
        # item = self.get_ret_item(item)
        image_id = int(self.image_ids[item])
        image_path = f"./data/coco/train2014/COCO_train2014_{image_id:012d}.jpg"
        if not os.path.isfile(image_path):
            image_path = f"./data/coco/val2014/COCO_val2014_{image_id:012d}.jpg"
        return tokens, mask, prefix, caption, image_path

    def __init__(self,  data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix: bool = False):
        super(ClipCocoDatasetWithImages, self).__init__(data_path, prefix_length, gpt2_type,
                                                        normalize_prefix=normalize_prefix)
        self.image_root = []
        self.images_names = []


def generate_beam(model: ClipCaptionModel, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    # beam_outputs = model.gpt.generate(max_length=50, num_beams=beam_size, no_repeat_ngram_size=2,
    #                                   num_return_sequences=5,
    #                                   inputs_embeds = embed,
    #                                   early_stopping=True, encoder_outputs=embed, temperature=1.)
    #
    # beam_outputs = [tokenizer.decode(item[embed.shape[1]:]) for item in beam_outputs]
    # return beam_outputs
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            #print(prompt)
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                #print("tokens")
                #print(type(tokens)) #torch.Size([1, 2])
                #print(tokens.size())
                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):
                # if generated.size(1) > 1024:
                #     print("DEBUG")
                #     print(len(prompt.split(' ')))
                #     print(len(prompt))
                #     print(generated.size())
                #     print(torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).size())
                #     generated = generated[:, :900]
                #     print(generated.size())

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                # next_token = torch.argmax()
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                # next_token = torch.multinomial(nnf.softmax(logits, dim=-1), num_samples=1)
                #print(next_token.size()) #torch.Size([1, 1])
                #print(generated.size()) #torch.Size([1, 2, 768])
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item() or next_token.item() == 764:
                    break


            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            # print("debug output")
            # print(output_text)
            # print(time.time() - start)
            generated_list.append(output_text)

    return generated_list[0]


def add_embedding_from_text(add_in: str, prefix_embed: T, tokenizer, model: ClipCaptionModel, where: int):
    device = prefix_embed.device
    tokens = torch.tensor(tokenizer.encode(add_in)).to(device)
    token_embedding = model.get_embedding(tokens).unsqueeze(0)
    if where == -1 or where == prefix_embed.shape[1]:
        prefix_list = (prefix_embed, token_embedding)
    elif where == 0:
        prefix_list = (token_embedding, prefix_embed)
    else:
        prefix_list = (prefix_embed[:, :where], token_embedding, prefix_embed[:, where:])
    prefix_new = torch.cat(prefix_list, dim=1)
    return prefix_new


def generate_text(prefix_embed: T, tokenizer, model: ClipCaptionModel, use_beam: bool) -> str:
    if use_beam:
        generated_text = generate_beam(model, tokenizer, embed=prefix_embed, beam_size=5)[0]
    else:
        generated_text = generate2(model, tokenizer, embed=prefix_embed)
    return generated_text


def re_caption(add_in : str, prefix_embed: T, tokenizer, model: ClipCaptionModel,
               where: int, use_beam: bool = True) -> str:
    prefix_new = add_embedding_from_text(add_in, prefix_embed, tokenizer, model, where)
    return generate_text(prefix_new, tokenizer, model, use_beam)


def remove_token(prefix_embed: T, tokenizer, model: ClipCaptionModel, embeddings,
                 where: List[int], use_beam: bool = True):
    prefix_new = [prefix_embed[:, i] for i in range(prefix_embed.shape[1]) if i not in where]
    prefix_new = torch.stack(prefix_new, dim=1)
    sim = torch.einsum('pd,nd->pn', nnf.normalize(prefix_new[0], 2, 1), embeddings)
    sim_arg = sim.argmax(-1)
    prefix_sent = tokenizer.decode(sim_arg)
    generated_text = generate_text(prefix_new, tokenizer, model, use_beam=use_beam)
    return generated_text, prefix_sent


def try_all_places(add_in : str, prefix_embed: T, tokenizer, model: ClipCaptionModel, use_beam: bool = True) -> List[str]:
    out = []
    for i in range(prefix_embed.shape[1]):
        out.append(re_caption(add_in, prefix_embed, tokenizer, model, i, use_beam))
    return out


def get_prefix_tokens(prefix_embed, embeddings, tokenizer) -> str:
    sim = torch.einsum('pd,nd->pn', nnf.normalize(prefix_embed[0], 2, 1), embeddings)
    sim_arg = sim.argmax(-1)
    prefix_tokens = tokenizer.decode(sim_arg)
    return prefix_tokens


def train(dataset: ClipCocoDataset, model: ClipCaptionModel, batch_size: int, device):
    model = model.to(device)
    model.eval()
    tokenizer = dataset.tokenizer
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    embeddings = model.gpt.get_input_embeddings().weight.data
    embeddings = nnf.normalize(embeddings, 2, 1)
    for idx, (tokens, mask, prefix, caption, images) in tqdm(enumerate(train_dataloader)):
        tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
        for jj in range(1, tokens.size(0)):
            found = False
            for item in ("19906", "320200", "341061", "400728", "444467"):
                if item in images[jj - 1]:
                    found = True
                    break
            if not found:
                continue
            prefix_embed = model.clip_project(prefix[jj - 1:jj]).reshape(1, dataset.prefix_length, -1)
            prefix_sent = get_prefix_tokens(prefix_embed, embeddings, tokenizer)
            try:
                generated_text_beam = generate_beam(model, tokenizer, embed=prefix_embed, beam_size=5)
                generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
            except BaseException:
                continue
                print("probability tensor contains either `inf`, `nan` or element < 0")
            if DEBUG:
                image_caption = f"\nGT: {caption[jj-1]}\n\nClipCap: {generated_text_prefix}"
                print(prefix_sent)
                imshow(images[jj - 1], image_caption)
            else:
                print("-=(%0d)=-" % jj)
                print("Caption:")
                print(caption[jj-1])
                print(">>>>> Generate from prefix")
                print(generated_text_beam[0])
        # user_input = input("\nto exit type x\n")
        # if user_input == "x":
        #     break
    return 0


def main():
    batch_size = 5
    num_epochs = 10
    prefix_length = 10
    model = ClipCaptionModel(prefix_length)
    device = CPU
    model.load_state_dict(torch.load("./checkpoints/oscar_split-007.pt", map_location=device))
    dataset = ClipCocoDatasetWithImages("./data/coco/oscar_split_train.pkl", prefix_length, normalize_prefix=False)

    # generated_text2 = generate_beam(model, GPT2Tokenizer.from_pretrained('gpt2'), prompt="Toronto Raptors")
    with torch.no_grad():
        train(dataset, model, batch_size, device)


if __name__ == '__main__':
    exit(main())

