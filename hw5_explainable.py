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
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor

"""Argument parsing"""
parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
parser.add_argument("--iterations", default=20, type=int, help="number of gradient ascent steps per octave")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--octave_scale", default=1.2, type=float, help="image scale between octaves")
parser.add_argument("--num_octaves", default=12, type=int, help="number of octaves")
parser.add_argument("--ckptpath", default="./checkpoint.pth")
parser.add_argument("--dataset_dir", default="./food-11/")
parser.add_argument("--split", default="evaluation")
parser.add_argument("--filename")
parser.add_argument("--layers", default="2,5,8,12,15,18,22,25,28,32")
parser.add_argument("--layerid", type=int)
parser.add_argument("--filterid", type=int)
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
def dream(image, model, iterations, lr, filterid=None):
    """ Updates the image to maximize outputs for n iterations """
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        if filterid is not None:
            out = out[:, filterid, :, :]
        loss = out.norm()
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()


def deep_dream(image, model, iterations, lr, octave_scale, num_octaves, filterid=None):
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
        dreamed_image = dream(input_image, model, iterations, lr, filterid=filterid)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)


if args.mode == 'dream':
    if args.filename is not None:
        input_image = os.path.join(args.dataset_dir, args.split, f'{args.filename}.jpg')
        image = Image.open(input_image)
    else:
        image = Image.fromarray((np.random.random((512, 512, 3)) * 255).astype(np.uint8))
    
    expdir = "dream"
    os.makedirs(expdir, exist_ok=True)

    if args.layerid is not None and args.filterid is not None:
        # dream specific filter
        all_layers = list(network.cnn.children())
        model = nn.Sequential(*all_layers[: (args.layerid + 1)])
        if torch.cuda.is_available:
            model = model.cuda()

        dreamed_image = deep_dream(
            image,
            model,
            iterations=args.iterations,
            lr=args.lr,
            octave_scale=args.octave_scale,
            num_octaves=args.num_octaves,
            filterid=args.filterid
        )

        activation = model(Tensor(dreamed_image).unsqueeze(0).permute(0, 3, 1, 2).cuda()).detach().cpu()[0, args.filterid]
        activation = nd.zoom(activation, dreamed_image.shape[0] / activation.shape[0])
        activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
        activation = torch.FloatTensor(activation)

        unit = 3
        fig, axs = plt.subplots(1, 2, figsize=(unit * 2, unit))
        axs[0].imshow(dreamed_image)
        axs[0].annotate(f"id {args.filterid}", (0, 0))
        axs[1].imshow(activation)
        axs[1].annotate(f"activate {activation.norm().item()}", (0, 0))
        plt.savefig(os.path.join(expdir, f"layer{args.layerid}_filter{args.filterid}.jpg"))

    elif args.layerid is not None:
        # dream all filters in the specified layer
        all_layers = list(network.cnn.children())
        model = nn.Sequential(*all_layers[: (args.layerid + 1)])
        if torch.cuda.is_available:
            model = model.cuda()
        
        with torch.no_grad():
            imgtmp = preprocess(image).unsqueeze(0).cpu().data.numpy()
            filter_num = model(Tensor(imgtmp)).size(1)

        unit = 3
        n_row = math.ceil(math.sqrt(filter_num))
        fig, axs = plt.subplots(n_row, 2 * n_row, figsize=(2 * n_row * unit, n_row * unit))
        axs = [axs[i, j] for i, j in itertools.product(range(n_row), range(2 * n_row))]

        for filterid in tqdm.tqdm(range(filter_num)):
            dreamed_image = deep_dream(
                image,
                model,
                iterations=args.iterations,
                lr=args.lr,
                octave_scale=args.octave_scale,
                num_octaves=args.num_octaves,
                filterid=filterid
            )

            activation = model(Tensor(dreamed_image).unsqueeze(0).permute(0, 3, 1, 2).cuda()).detach().cpu()[0, filterid]
            activation = nd.zoom(activation, dreamed_image.shape[0] / activation.shape[0])
            activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
            activation = torch.FloatTensor(activation)

            axs[2*filterid].imshow(dreamed_image)
            axs[2*filterid].annotate(f"id {filterid}", (0, 0))
            axs[2*filterid+1].imshow(activation)
            axs[2*filterid+1].annotate(f"activate {activation.norm().item()}", (0, 0))

        plt.savefig(f"{expdir}/filters_layer{args.layerid}.jpg")
        
    else:
        # dream all layers
        unit = 10
        layers = [int(idx) for idx in args.layers.split(',')]
        n_row = math.ceil(math.sqrt(len(layers) + 1))
        fig, axs = plt.subplots(n_row, n_row, figsize=(n_row * unit, n_row * unit))
        axs = [axs[i, j] for i, j in itertools.product(range(n_row), range(n_row))]
        axs[0].imshow(np.array(image))

        for idx, layerid in enumerate(layers):
            print(f"Processing layer {layerid}")
            all_layers = list(network.cnn.children())
            model = nn.Sequential(*all_layers[: (layerid + 1)])
            if torch.cuda.is_available:
                model = model.cuda()

            dreamed_image = deep_dream(
                image,
                model,
                iterations=args.iterations,
                lr=args.lr,
                octave_scale=args.octave_scale,
                num_octaves=args.num_octaves,
            )

            axs[idx + 1].imshow(dreamed_image)
            axs[idx + 1].annotate(layerid, (0, 0))

        plt.savefig(f"{expdir}/layers.jpg")
        plt.close()

elif args.mode == 'activation':
    assert args.layerid is not None
    assert args.filterid is not None

    all_layers = list(network.cnn.children())
    model = nn.Sequential(*all_layers[: (args.layerid + 1)])
    if torch.cuda.is_available:
        model = model.cuda()
    
    activations = []
    for img, label in tqdm.tqdm(dataset):
        out = model(img.unsqueeze(0).cuda())
        activation = out[:, args.filterid, :, :][0].detach().cpu()
        activations.append((img, activation, activation.norm().item()))
    
    activations_sorted = sorted(activations, key=lambda x: x[2], reverse=True)
    
    unit = 3
    n_row = 10
    fig, axs = plt.subplots(n_row, 2 * n_row, figsize=(2 * n_row * unit, n_row * unit))
    axs = [axs[i, j] for i, j in itertools.product(range(n_row), range(2 * n_row))]
    for i in range(n_row * n_row):
        img = activations_sorted[i][0].permute(1, 2, 0)
        activation = activations_sorted[i][1]
        activation = torch.FloatTensor(nd.zoom(activation.numpy(), img.size(0) / activation.size(0)))
        activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
        
        axs[2*i].imshow(img)
        axs[2*i+1].imshow(activation)
        axs[2*i+1].annotate(activations_sorted[i][2], (0, 0))

    expdir = 'activations'
    os.makedirs(expdir, exist_ok=True)
    plt.savefig(os.path.join(expdir, f'layer{args.layerid}_filter{args.filterid}.jpg'))