import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class ResNet(nn.Module):
        def __init__ (self, n_layers, block, num_classes = 10):
                super(ResNet, self).__init__()
                self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
                self.bn1 = nn.BatchNorm2d(16)
                self.relu = nn.ReLU(inplace = True)
                
                self.layers1 = self._make_layers(block, 16, 16, stride = 1)
                self.layers2 = self._make_layers(block, 16, 32, stride = 2)
                self.layers3 = self._make_layers(block, 32, 64, stride = 2)

                self.avg_pooling = nn.AvgPool2d(8, stride = 1)
                self.fc_out = nn.Linear(64, num_classes)

                #weight_initialization
                for m in self.modules(): 
                        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                                init.kaiming_normal_(m.weight)

        def _make_layers(self, block, in_channels, out_channels, stride, n = 5):

                if stride == 2:
                        down_sample = True
                else: down_sample = False

                layers = nn.ModuleList([block(in_channels, out_channels, down_sample)])

                for _ in range(n - 1):
                        layers.append(block(out_channels, out_channels))

                return nn.Sequential(*layers)

        def forward(self, x):

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.layers1(out)
                out = self.layers2(out)
                out = self.layers3(out)

                out = self.avg_pooling(out)
                out = out.view(out.size(0), -1)
                out = self.fc_out(out)

                return out

class ResidualBlock(nn.Module):

        def __init__(self, in_channels, out_channels, stride = 1, down_sample = False):
                super(ResidualBlock, self).__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False )
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace = True)

                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.stride = stride

                # downsampling (dotted line)
                if down_sample:
                        self.down_sample = IdentityPadding(in_channels, out_channels, stride)
                else:
                        self.down_sample = None 

        def forward(self, x):

                shortcut = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.down_sample is not None:
                        shortcut = self.down_sample(x)

                out = out + shortcut 
                out = self.relu(out)

                return out 

# Downsampling Option A
class IdentityPadding(nn.Module):

        def __init__(self, in_channels, out_channels, stride):
                super(IdentityPadding, self).__init__()

                self.pooling = nn.MaxPool2d(1, stride=stride)
                self.fill_channel = out_channels - in_channels

        def forward(self, x):

                out = F.pad(x, (0, 0, 0, 0, 0, self.fill_channel))
                out = self.pooling(out)
                return out 

def resnet():
        return ResNet(5, block = ResidualBlock)


