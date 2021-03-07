import torch
import argparser
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from train import train
from resnet import Resnet
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

######################################################
parser = argparse.ArgumentParser(description = 'Pytorch Resnet-32 model for CIFAR-10 Classification')

parser.add_argument('--lr', default=0.1, help = 'learning rate')
parser.add_argument('--weight_decay', default = 1e-4, help = 'weight_decay')
parser.add_argument('--momentum', default = 0.9 help = 'momentum')

parser.add_argument('--Epoch', default = 80, help = ' Epoch ')
parser.add_argument('--batch_size', default = 128, help = 'TRAIN batch size ')
parser.add_argument('--test_batch_size', default = 100, help = 'TEST batch size')
parser.add_argument('--num_worker', default = 4, help = 'Num of workers')
parser.add_argument('--logdir', type = str, default = 'logs', help = '')

args = parser.parse_args()
######################################################
