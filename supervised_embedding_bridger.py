import argparse
import torch
import wandb
import pickle, math, random
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

#  The purpose here is to learn a fully supervised mapper from the image embeddings to the text embeddings,
#  using a simple MLP with ReLU activation with L2 loss. Then we can use this mapper to bridge the modality gap.


DBG = False
BTCH_SIZE = 32
PATH = 'weights_modality_mapper.pt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_map_to_text_space_using_modality_bridger():
    # load model
    model = MLP(640, 640, 640, 3).to(device)
    model.load_state_dict(torch.load('others/' + PATH))
    model.eval()

    def map_to_text_space_using_modality_bridger(image_embedding):
        return model(image_embedding)

    return map_to_text_space_using_modality_bridger


def get_dataloader():
    with open('coco_clip_embeddings/oscar_split_RN50x4_train_with_text_embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
        images_embeddings = data['clip_embedding'][:320 if DBG else -1].float()
        images_embeddings /= torch.norm(images_embeddings, dim=1, keepdim=True)

        text_embeddings = data['clip_embedding_text_dave'][:320 if DBG else -1].float()
        text_embeddings /= torch.norm(text_embeddings, dim=1, keepdim=True)
    return get_torch_dataloader_from_list(images_embeddings, text_embeddings, batch_size_train=BTCH_SIZE,
                                   batch_size_test=BTCH_SIZE,
                                   split=[int(0.8 * images_embeddings.shape[0]), int(0.2 * images_embeddings.shape[0])])


def get_torch_dataloader_from_list(list_x, list_y, list_x_test=None, list_y_test=None, batch_size_train=1, batch_size_test=1, split=None):
    my_dataset = torch.utils.data.TensorDataset(torch.Tensor(list_x.float()), torch.Tensor(list_y).float())
    my_dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size_train)  # create your dataloader
    if split is not None:
        assert list_x_test is None and list_y_test is None
        split[0] += (len(list_x) - (split[0] + split[1]))
        train_set, val_set = torch.utils.data.random_split(my_dataset, split)
        return torch.utils.data.DataLoader(train_set, batch_size=batch_size_train), torch.utils.data.DataLoader(val_set, batch_size=batch_size_test)

    if list_x_test is not None and list_y_test is not None:
        my_dataset_test = torch.utils.data.TensorDataset(torch.Tensor(list_x_test), torch.Tensor(list_y_test))  # create your dataset
        my_dataloader_test = torch.utils.data.DataLoader(my_dataset_test, batch_size=batch_size_test)  # create your dataloader
        return my_dataloader, my_dataloader_test

    return my_dataloader


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return torch.sigmoid(x)


def noise_augmentation(x, variance=0.001):
    if variance == 0.0:
        return x
    x = torch.nn.functional.normalize(x, dim=1)
    x = x + (torch.randn(x.shape, device=device) * math.sqrt(variance))
    return torch.nn.functional.normalize(x, dim=1)


def train_modal():
    dl_train, dl_test = get_dataloader()

    model = MLP(640, 640, 640, 3).to(device)
    wandb.init(project='bridger', entity='ml_lab')
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(2 if DBG else 8):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0.0
        for i, data in enumerate(dl_train, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device).view(-1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(inputs).view(-1)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0 and i != 0:  # print every 2000 mini-
                wandb.log({'epoch': epoch, 'train_loss': running_loss / 100})
                running_loss = 0.0

        # valid set
        running_loss = 0.0
        for i, data in enumerate(dl_test, 0):
            inputs, labels = data[0].to(device), data[1].to(device).view(-1)
            # forward + backward + optimize
            with torch.no_grad():
                output = model(inputs).view(-1)
                loss = criterion(output, labels)
            running_loss += loss.item()
            if i % 100 == 0 and i != 0:  # print every 2000 mini-
                wandb.log({'epoch': epoch, 'val_loss': running_loss / 100})
                running_loss = 0.0

    print('Finished Training')
    torch.save(model.state_dict(), PATH)
    return -1


if "__main__" == __name__:
    train_modal()

