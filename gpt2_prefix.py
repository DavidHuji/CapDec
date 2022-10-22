from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
from custom_types import *
import pickle
import sys
import argparse
import json
import time
import transformer_mapper
import clip


class MappingType(Enum):
    MLP = 'mlp'
    TransformerEncoder = 'transformer_encoder'
    TransformerDecoder = 'transformer_decoder'


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def get_ret_item(self, item):
        return self.my_ret[item % len(self.my_ret)]

    def __getitem__(self, item: int) -> TS:
        # item = self.get_ret_item(item)
        tokens, mask = self.pad_tokens(item)
        prefix: T = self.prefixes[ self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix, self.captions[item]

    @staticmethod
    def add_period(captions_raw):
        for item in captions_raw:
            caption = item['caption']
            caption = caption.strip()
            if caption[-1] != '.':
                item['caption'] = caption + '.'
            elif caption[-2] == ' ':
                item['caption'] = caption[:-2] + '.'
        return captions_raw

    @staticmethod
    def get_tokenizer(gpt2_type, num_trials: int = 100) -> Optional[GPT2Tokenizer]:
        tokenizer = None
        for i in range(num_trials):
            try:
                tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
                break
            except ValueError:
                time.sleep(1)
        if tokenizer is None:
            raise ValueError
        return tokenizer

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2", normalize_prefix: bool = False):
        self.tokenizer = self.get_tokenizer(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print(data_path)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        print("Data size is %0d" % len(all_data["captions"]))
        sys.stdout.flush()
        self.prefixes: T = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        captions_raw = self.add_period(captions_raw)
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for i, caption in enumerate(captions_raw):
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                 pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        # all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        # self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        self.max_seq_len = 40
        # self.my_ret = [i for i in range(len(self)) if self.image_ids[i] in ("19906", "320200", "341061", "400728", "444467") ]




class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MappingNetwork(nn.Module):

    def forward(self, x):
        return self.mlp(x)

    def __init__(self, prefix_dim: int, prefix_length: int, embedding_dim: int):
        super(MappingNetwork, self).__init__()
        self.mlp = MLP(tuple([prefix_dim] * 7 + [prefix_length * embedding_dim]), act=nn.LeakyReLU)


class ClipCaptionModel(nn.Module):

    #@functools.lru_cache
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        #print(embedding_text.size()) #torch.Size([5, 67, 768])
        #print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None,
                 prefix_dim: int = 640, num_layers: int = 8, mapping_type: MappingType = MappingType.TransformerEncoder):
        super(ClipCaptionModel, self).__init__()
        clip_length = prefix_length if clip_length is None else clip_length
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.TransformerEncoder:
            self.clip_project = transformer_mapper.TransformerMapper(prefix_dim, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)
        elif mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_dim, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = transformer_mapper.TransformerEncoderDecoder(prefix_dim, self.gpt_embedding_size,
                                                                             prefix_length, clip_length, num_layers)

        # self.clip_project = transformer_mapper.TransformerEncoderDecoder(prefix_dim, self.gpt_embedding_size,
        #                                                                  prefix_length, num_layers)
        # self.clip_project = MLP((prefix_dim, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=CPU))
    else:
        print(f"{model_path} is not exist")
    return model, parser


def train(dataset: ClipCocoDataset, model: ClipCaptionModel, batch_size: int,
          epochs: int, lr: float=2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = "",
          save_model_on_epoch: bool = True, args=None):
    device = CUDA(0)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix, _) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
        else:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}_latest.pt"),
            )

    return model


def create_few(data_path, num_samples, out_file: str):
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    clip_embedding = all_data["clip_embedding"]
    captions = all_data["captions"]
    select = torch.rand(len(captions)).argsort()[:num_samples]
    clip_embedding_new = clip_embedding[select]
    captions_new = [captions[i] for i in select]
    for i in range(len(captions_new)):
        captions_new[i]['clip_embedding'] = i
    with open(out_file, 'wb') as f:
        pickle.dump({"captions": captions_new, "clip_embedding": clip_embedding_new}, f)


def main():
    for i in (80, ):
        MLP = 'mlp'
        TransformerEncoder = 'transformer_encoder'
        TransformerDecoder = 'transformer_decoder'
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', default='./data/conceptual/conceptual_clip_train_rn.pkl')
        parser.add_argument('--out_dir', default='./checkpoints')
        parser.add_argument('--prefix', default=f'conceptual_prefix_td10_{i}_rn')
        parser.add_argument('--epochs', type=int, default=20)
        parser.add_argument('--save_every', type=int, default=5)
        parser.add_argument('--prefix_length', type=int, default=i)
        parser.add_argument('--prefix_length_clip', type=int, default=10)
        parser.add_argument('--mapping_type', type=str, default='transformer_decoder',
                            help='mlp/transformer_encoder/transformer_decoder')
        parser.add_argument('--bs', type=int, default=24)
        parser.add_argument('--only_prefix', dest='', action='store_true')
        parser.add_argument('--num_layers', type=int, default=4)
        parser.set_defaults(only_prefix=False)
        args = parser.parse_args()
        args.only_prefix = True
        batch_size = args.bs
        num_epochs = args.epochs
        prefix_length = args.prefix_length
        # create_few(args.data, 1000, './data/coco/oscar_split_train_1000.pkl')
        prefix_dim = [512, 640]['rn' in args.data.lower()]
        args.mapping_type = {'mlp': MappingType.MLP, 'transformer_encoder': MappingType.TransformerEncoder,
                             'transformer_decoder': MappingType.TransformerDecoder}[args.mapping_type]
        if args.only_prefix:
            model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_dim=prefix_dim,
                                      num_layers=args.num_layers, mapping_type=args.mapping_type)
            print("Train only prefix")
        else:
            model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_dim=prefix_dim,
                                     num_layers=args.num_layers, mapping_type=args.mapping_type)
            print("Train both prefix and GPT")
            sys.stdout.flush()
        num_params = sum([torch.numel(param) for param in model.parameters()])
        print(f"num_params: {num_params}")
        dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=True)
        train(dataset, model, batch_size, num_epochs, output_dir=args.out_dir, output_prefix=args.prefix, args=args)


if __name__ == '__main__':
    main()
