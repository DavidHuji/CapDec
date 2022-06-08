import sys
sys.path.append("/home/amir/projects/CLIP")
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import clip   # installed from https://github.com/openai/CLIP
from gpt2_prefix import ClipCocoDataset, MLP
from custom_types import *
from PIL import Image
import sys
from gpt2_prefix_eval import generate2, imshow, generate_beam
import pickle
from pycocotools.coco import COCO


class ClipCocoDatasetWithImages(ClipCocoDataset):

    def __getitem__(self, item):
        tokens, mask, _, caption = super(ClipCocoDatasetWithImages, self).__getitem__(item)
        image_path = f"{self.image_root}/{self.images_names[self.caption2embedding[item]]}"
        image = self.transform(Image.open(image_path).convert("RGB"))
        return tokens, mask, image, caption, image_path

    def __init__(self, ann_suffix: str, data_path: str,  transform):
        super(ClipCocoDatasetWithImages, self).__init__(data_path, 10, "gpt2")
        ann_file = f"./data/coco/annotations/captions_{ann_suffix}.json"
        # with open(f"./data/coco/oscar_split_train_images.pkl", 'rb') as f:
        #     self.images_names = pickle.load(f)
        self.image_root = f'./data/coco/{ann_suffix}'
        self.transform = transform
        coco = COCO(ann_file)
        img_ids = coco.getImgIds()
        self.images_names = list(map(lambda x: x['file_name'], coco.loadImgs(img_ids)))


class ClipCaptionE2E(nn.Module):

    def forward(self, tokens: T, image: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix = self.forward_image(image)
        embedding_cat = torch.cat((prefix, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def forward_image(self, x: T):
        x = self.clip_model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.positional_embedding.to(x.dtype)
        x = self.clip_model.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)[:, : self.prefix_length]
        prefix = self.clip_project(x)
        return prefix

    def __init__(self, prefix_size: int = 768):
        super(ClipCaptionE2E, self).__init__()
        self.prefix_length = 10
        clip_model, self.transform = clip.load("ViT-B/32", device=CPU, jit=False)
        self.clip_model = clip_model.visual
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size, prefix_size))


def train(dataset: ClipCocoDataset, model: ClipCaptionE2E, batch_size: int,
          epochs: int, lr: float=2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = "",
          save_model_on_epoch: bool = True):
    device = CUDA(0)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=0 if DEBUG else 4)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader) * 2
    )
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, image, _, _) in enumerate(train_dataloader):

            model.zero_grad()
            tokens, mask, image = tokens.to(device), mask.to(device), image.to(device, dtype=torch.float32)
            outputs = model(tokens, image, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        if save_model_on_epoch or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )

    return model


def evaluate(dataset: ClipCocoDatasetWithImages, model: ClipCaptionE2E, batch_size: int, device):
    model = model.to(device)
    model.eval()
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)


    for idx, (tokens, mask, images, caption, image_path) in tqdm(enumerate(train_dataloader)):

        tokens, mask, images = tokens.to(device), mask.to(device), images.to(device)

        for jj in range(1, tokens.size(0)):
            prefix_embed = model.forward_image(images[jj - 1:jj])
            try:
                generated_text_prefix = generate_beam(model, dataset.tokenizer, embed=prefix_embed)
            except RuntimeError:
                print("probability tensor contains either `inf`, `nan` or element < 0")
            if DEBUG:
                image_caption = f"\nGT: {caption[jj-1]}\n\nPrefix: {generated_text_prefix[0]}"
                imshow(image_path[jj - 1], image_caption)
            else:
                print("-=(%0d)=-" % jj)
                print("Caption:")
                print(caption[jj-1])
                print(">>>>> Generate from prefix")
                print(generated_text_prefix)
        user_input = input("\nto exit type x\n")
        if user_input == "x":
            break
    return 0


def main():
    batch_size = 20
    num_epochs = 10
    model = ClipCaptionE2E()
    transform = model.transform
    # model = nn.DataParallel(model)

    dataset = ClipCocoDatasetWithImages("train2014", "./data/coco/oscar_split_train.pkl", transform)
    train(dataset, model, batch_size, num_epochs, output_dir="./checkpoints", output_prefix="coco_e2e")


def evaluate_main():
    model = ClipCaptionE2E()
    device = CUDA(3)
    model.load_state_dict(torch.load("./checkpoints/coco_e2e_latest.pt", map_location=device))
    dataset = ClipCocoDatasetWithImages("val2014", "./data/coco/coco_clip_val2014.pkl", model.transform)
    evaluate(dataset, model, 5, device)


if __name__ == '__main__':
    evaluate_main()
