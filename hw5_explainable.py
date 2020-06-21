# -*- coding: utf-8 -*-

import os
import sys
import math
import argparse
import itertools
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from ipdb import set_trace
from torch.autograd import Variable
import tqdm
import scipy.ndimage as nd
from utils import deprocess, preprocess, clip

"""Argument parsing"""
parser = argparse.ArgumentParser()
parser.add_argument("--iterations", default=20, type=int, help="number of gradient ascent steps per octave")
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--octave_scale", default=1.4, type=float, help="image scale between octaves")
parser.add_argument("--num_octaves", default=10, type=int, help="number of octaves")
parser.add_argument("--ckptpath", default="./checkpoint.pth")
parser.add_argument("--dataset_dir", default="./food-11/")
parser.add_argument("--filename", default="0-0")
parser.add_argument("--split", default="evaluation")
parser.add_argument("--from_noise", action="store_true")
parser.add_argument("--layers", default="2,5,8,12,15,18,22,25,28,32")
args = parser.parse_args()


"""Model definition and checkpoint loading"""
class Classifier(nn.Module):
  def __init__(self):
    super(Classifier, self).__init__()

    def building_block(indim, outdim):
      return [
        nn.Conv2d(indim, outdim, 3, 1, 1),
        nn.BatchNorm2d(outdim),
        nn.ReLU(),
      ]
    def stack_blocks(indim, outdim, block_num):
      layers = building_block(indim, outdim)
      for i in range(block_num - 1):
        layers += building_block(outdim, outdim)
      layers.append(nn.MaxPool2d(2, 2, 0))
      return layers

    cnn_list = []
    cnn_list += stack_blocks(3, 128, 3)
    cnn_list += stack_blocks(128, 128, 3)
    cnn_list += stack_blocks(128, 256, 3)
    cnn_list += stack_blocks(256, 512, 1)
    cnn_list += stack_blocks(512, 512, 1)
    self.cnn = nn.Sequential( * cnn_list)

    dnn_list = [
      nn.Linear(512 * 4 * 4, 1024),
      nn.ReLU(),
      nn.Dropout(p = 0.3),
      nn.Linear(1024, 11),
    ]
    self.fc = nn.Sequential( * dnn_list)

  def forward(self, x):
    out = self.cnn(x)
    out = out.view(out.size()[0], -1)
    return self.fc(out)

network = Classifier().cuda()
checkpoint = torch.load(args.ckptpath)
network.load_state_dict(checkpoint['model_state_dict'])


"""Dataset definition and creation"""
class FoodDataset(Dataset):
    def __init__(self, paths, labels, mode):
        # mode: 'train' or 'eval'

        self.paths = paths
        self.labels = labels
        trainTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        evalTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
        ])
        self.transform = trainTransform if mode == 'train' else evalTransform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = Image.open(self.paths[index])
        X = self.transform(X)
        Y = self.labels[index]
        return X, Y

    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
          image, label = self.__getitem__(index)
          images.append(image)
          labels.append(label)
        return torch.stack(images), torch.tensor(labels)

def get_paths_labels(path):
    imgnames = os.listdir(path)
    imgnames.sort()
    imgpaths = []
    labels = []
    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))
    return imgpaths, labels
paths, labels = get_paths_labels(os.path.join(args.dataset_dir, args.split))
dataset = FoodDataset(paths, labels, mode='eval')


"""Filter explaination"""
def dream(image, model, iterations, lr):
    """ Updates the image to maximize outputs for n iterations """
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss = out.norm()
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()


def deep_dream(image, model, iterations, lr, octave_scale, num_octaves):
    """ Main deep dream method """
    image = preprocess(image).unsqueeze(0).cpu().data.numpy()

    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)


# Load image
if not args.from_noise:
    expdir = f"outputs/{args.filename}"
    input_image = os.path.join(args.dataset_dir, args.split, f'{args.filename}.jpg')
    image = Image.open(input_image)
else:
    expdir = f"outputs/noise"
    image = Image.fromarray((np.random.random((512, 512, 3)) * 255).astype(np.uint8))
os.makedirs(expdir, exist_ok=True)

layers = [int(idx) for idx in args.layers.split(',')]
n_grid = len(layers) + 1
n_row = math.ceil(math.sqrt(n_grid))
fig, axs = plt.subplots(n_row, n_row, figsize=(n_row * 10, n_row * 10))
axs = [axs[i, j] for i, j in itertools.product(range(n_row), range(n_row))]
axs[0].imshow(np.array(image))

for idx, at_layer in enumerate(layers):
    print(f"Processing layer {at_layer}")
    all_layers = list(network.cnn.children())
    model = nn.Sequential(*all_layers[: (at_layer + 1)])
    if torch.cuda.is_available:
        model = model.cuda()

    # Extract deep dream image
    dreamed_image = deep_dream(
        image,
        model,
        iterations=args.iterations,
        lr=args.lr,
        octave_scale=args.octave_scale,
        num_octaves=args.num_octaves,
    )

    axs[idx + 1].imshow(dreamed_image)

plt.savefig(f"{expdir}/summary.jpg")
plt.close()
