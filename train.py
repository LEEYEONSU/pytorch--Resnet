import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import resnet

# CIFAR-10 training transformation
transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()
])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root = './data/', 
                                                                        train = True, 
                                                                        transform = transform, 
                                                                        download = True)
test_dataset = torchvision.datasets.CIFAR10(root = './data/', 
                                                                        train = False, 
                                                                        transform = transforms.ToTensor(), 
                                                                        download = True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100, 
                                          shuffle=False)

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Epoch = 80, lr = 1e-3