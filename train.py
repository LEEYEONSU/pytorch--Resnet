import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import resnet

# CIFAR-10 training transformation
transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
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

# weight_init? reference 달림 
# optimizer weight_decay = 1e-4, momentum = 0.9
# Epoch = 80, lr = MultiStepLR 
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])