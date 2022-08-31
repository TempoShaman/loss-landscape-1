import os
import sys
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from few_shot.datasets import MiniImageNet
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from cifar10.datasets import MiniImageNet
import random
import itertools
import collections
DATA_PATH = "/home/oliver/Documents/loss-landscape/cifar10/data"
import glob
import model_loader
from torch.autograd.variable import Variable
import cv2
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
import skimage
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import torch.utils.data as data

#from dataset import Omniglot, MNIST


class FewShotDataset(data.Dataset):
    """
    Load image-label pairs from a task to pass to Torch DataLoader
    Tasks consist of data and labels split into train / val splits
    """

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform  # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.root = self.task.root
        self.split = split
        self.img_ids = self.task.train_ids if self.split == 'train' else self.task.val_ids
        self.labels = self.task.train_labels if self.split == 'train' else self.task.val_labels

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

def get_relative_path(file):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    return os.path.join(script_dir, file)

class Omniglot(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def load_image(self, idx):
        ''' Load image '''
        im = Image.open('{}/{}'.format(self.root, idx)).convert('RGB')
        im = im.resize((28, 28), resample=Image.LANCZOS)  # per Chelsea's implementation
        im = np.array(im, dtype=np.float32)
        # print('000000000000000000000000000')
        # print(im.shape)
        # print('000000000000000000000000000')
        return im

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        im = self.load_image(img_id)
        if self.transform is not None:
            im = self.transform(im)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        # print('+++++++++++++++++++++')
        # print('+++++++++++++++++++++')
        # print(im.size())
        # print(label)
        # print('+++++++++++++++++++++')
        # print('+++++++++++++++++++++')
        return im, label


class MNIST(data.Dataset):

    def __init__(self, *args, **kwargs):
        super(MNIST, self).__init__(*args, **kwargs)

    def load_image(self, idx):
        ''' Load image '''
        # NOTE: we use the PNG dataset because meta-learning results in an error
        # when using the bitmap dataset and PyTorch unpacker
        im = Image.open('{}/{}.png'.format(self.root, idx)).convert('RGB')
        im = np.array(im, dtype=np.float32)
        print('+++++++++++++++++++++')
        print('+++++++++++++++++++++')
        print(im.size())
        print('+++++++++++++++++++++')
        print('+++++++++++++++++++++')
        return im

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = self.load_image(img_id)
        if self.transform is not None:
            img = self.transform(img)
        target = self.labels[idx]
        if self.target_transform is not None:
            target = self.target_transform(target)


        return img, target



class ClassBalancedSampler(Sampler):
    '''
    Samples class-balanced batches from 'num_cl' pools each
    of size 'num_inst'
    If 'batch_cutoff' is None, indices for iterating over batches
    of the entire dataset will be returned
    Otherwise, indices for the number of batches up to the batch_cutoff
    will be returned
    (This is to allow sampling with replacement across training iterations)
    '''

    def __init__(self, num_cl, num_inst, batch_cutoff=None):
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.batch_cutoff = batch_cutoff

    def __iter__(self):
        '''return a single list of indices, assuming that items will be grouped by class '''
        # First construct batches of 1 instance per class
        batches = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
        batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]
        # Shuffle within each batch so that classes don't always appear in same order
        for sublist in batches:
            random.shuffle(sublist)

        if self.batch_cutoff is not None:
            random.shuffle(batches)
            batches = batches[:self.batch_cutoff]

        batches = [item for sublist in batches for item in sublist]

        return iter(batches)

    def __len__(self):
        return 1


def get_data_loader(task, batch_size=1, split='train'):
    # NOTE: batch size here is # instances PER CLASS
    if task == 'mnist':
        normalize = transforms.Normalize(mean=[0.13066, 0.13066, 0.13066], std=[0.30131, 0.30131, 0.30131])
        dset = MNIST(task, transform=transforms.Compose([transforms.ToTensor(), normalize]), split=split)
    else:
        normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
        dset = Omniglot(task, transform=transforms.Compose([transforms.ToTensor(), normalize]), split=split)
    # else:
    #     normalize = transforms.Normalize(mean=[0.92206], std=[0.08426])
    #     dset = Omniglot(task, transform=transforms.Compose([transforms.ToTensor(), normalize]), split=split)


    sampler = ClassBalancedSampler(task.num_cl, task.num_inst, batch_cutoff = (None if split != 'train' else batch_size))
    loader = DataLoader(dset, batch_size=batch_size*task.num_cl, sampler=sampler, num_workers=1, pin_memory=True)
    return loader



###############################################################
####                        MAIN
###############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--dataset', default='miniimagenet', help='cifar10 | imagenet')
    parser.add_argument('--datapath', default='cifar10/data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='do not normalize data')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    args = parser.parse_args()

    #n_epochs = 80
    #dataset_class = MiniImageNet
    #num_input_channels = 3
    #drop_lr_every = 40

    #trainloader, testloader = background(), evaluation()

