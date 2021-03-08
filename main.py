import os 
import time
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

from utils.train import *
from model.resnet import resnet
from utils.function import *
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

#. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#

parser = argparse.ArgumentParser(description = 'Pytorch Resnet-32 model for CIFAR-10 Classification')

parser.add_argument('--print-freq', default=10, type=int, help = 'Print frequency')
parser.add_argument('--save_dir', default='./save_model/', type=str, help = 'saving model path')
parser.add_argument('--save-every', dest='save_every',help='Saves checkpoints at every specified number of epochs', type=int, default=10)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

parser.add_argument('--lr', default=0.1, help = 'learning rate')
parser.add_argument('--weight_decay', default = 1e-4, help = 'weight_decay')
parser.add_argument('--momentum', default = 0.9, help = 'momentum')

parser.add_argument('--Epoch', default = 80, help = ' Epoch ')
parser.add_argument('--batch_size', default = 128, help = 'TRAIN batch size ')
parser.add_argument('--test_batch_size', default = 100, help = 'TEST batch size')
parser.add_argument('--num_worker', default = 4, help = 'Num of workers')
parser.add_argument('--logdir', type = str, default = 'logs', help = '')

args = parser.parse_args()

#. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .#

if __name__ == '__main__':
    
        main(args)
        