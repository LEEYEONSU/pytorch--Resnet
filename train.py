import os 
import time
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

from resnet import resnet
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


######################################################
parser = argparse.ArgumentParser(description = 'Pytorch Resnet-32 model for CIFAR-10 Classification')

parser.add_argument('--print-freq', default=30, type=int, help = 'Print frequency')
parser.add_argument('--save_dir', default='./save_model/', type=str, help = 'saving model path')

parser.add_argument('--lr', default=0.1, help = 'learning rate')
parser.add_argument('--weight_decay', default = 1e-4, help = 'weight_decay')
parser.add_argument('--momentum', default = 0.9, help = 'momentum')

parser.add_argument('--Epoch', default = 80, help = ' Epoch ')
parser.add_argument('--batch_size', default = 128, help = 'TRAIN batch size ')
parser.add_argument('--test_batch_size', default = 100, help = 'TEST batch size')
parser.add_argument('--num_worker', default = 4, help = 'Num of workers')
parser.add_argument('--logdir', type = str, default = 'logs', help = '')

args = parser.parse_args()
######################################################

def main():
        # CIFAR-10 Training & Test Transformation
        print('. . . . . . . . . . . . . . . .PREPROCESSING DATA . . . . . . . . . . . . . . . .')
        TRAIN_transform = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        VAL_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),      
        ])

        # CIFAR-10 dataset
        train_dataset = torchvision.datasets.CIFAR10(root = '../data/', 
                                                                                train = True, 
                                                                                transform = TRAIN_transform, 
                                                                                download = True)
        val_dataset = torchvision.datasets.CIFAR10(root = '../data/', 
                                                                                train = False, 
                                                                                transform = VAL_transform,
                                                                                download = True)

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size = args.batch_size , 
                                                shuffle=True)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size = args.test_batch_size , 
                                                shuffle=False)

        # Device Config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = resnet()
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters() , lr = args.lr , weight_decay = args.weight_decay, momentum = args.momentum)
        lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones = [32000,48000], gamma = 0.1)

        #  Epoch = args.Epoch
        for epoch in range(0, args.Epoch):

                print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
                train(train_loader, model, criterion, optimizer, args.Epoch)
                lr_schedule.step()

                prec1 = validate(val_loader, model, criterion)

                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)

                if epoch > 0 and epoch % args.save_every == 0:
                        save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

                save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

def train(train_loader, model, criterion, optimizer, epoch):
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        model.train()

        end = time.time()

        for i, (input_, target) in enumerate(train_loader):

                data_time.update(time.time() - end)

                input_v = input_.cuda()
                target = target.cuda()
                target_v = target

                output = model(input_v)
                loss = criterion(output, target_v)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                output = output.float()
                loss = loss.float()
                # measure accuracy and record loss
                prec1 = accuracy(output.data, target)[0]
                losses.update(loss.item(), input_.size(0))
                top1.update(prec1.item(), input_.size(0))

                # elapsed time
                batch_time.update( time.time() - end )
                end = time.time()

                if i % args.print_freq == 0:
                        print('Epoch: [{0}][{1}/{2}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                                epoch, i, len(train_loader), batch_time=batch_time,
                                data_time=data_time, loss=losses, top1=top1))

def validation(val_loader, model, criterion):

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        model.eval()

        end = time.time()

        with torch.no_grad():
                for i, (input_, target) in enumerate(val_loader):
                        input_v = input_.cuda()
                        target = target.cuda()
                        target_v = target

                        output = model(input_v)
                        loss = loss.float()

                        prec1 = accuracy(output.data, target)[0]
                        losses.update(loss.item(), input_.size(0))
                        top1.update(prec1.item(), input_.size(0))

                        # measure elapsed time
                        batch_time.update(time.time() - end)
                        end = time.time()

                        if i % args.print_freq == 0:
                                rint('Test: [{0}/{1}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                                        i, len(val_loader), batch_time=batch_time, loss=losses,
                                        top1=top1))

                print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

        return top1.avg

### CODE FROM https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/trainer.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

main()