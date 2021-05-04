import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# Resnet-34 
class ResNet34(nn.Module):
        def __init__ (self, num_blocks, block, num_classes = 1000):
                super(ResNet, self).__init__()
                self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace = True)
                
                self.layers1 = self._make_layers(block, 64, 64, stride = 1, n = num_blocks[0])
                self.layers2 = self._make_layers(block, 64, 128, stride = 2, n = num_blocks[1])
                self.layers3 = self._make_layers(block, 128, 256, stride = 2, n = num_blocks[2])
                self.layers4 = self._make_layers(block, 256, 512, stride = 2, n = num_blocks[3])

                self.avg_pooling = nn.AvgPool2d(4, stride = 1)
                self.fc_out = nn.Linear(512, num_classes)

                #weight_initialization
                for m in self.modules(): 
                        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                                init.kaiming_normal_(m.weight)
                        elif isinstance(m, nn.BatchNorm2d):
                                init.constant_(m.weight, 1)
                                init.constant_(m.bias, 0)

        # n = # of layers
        def _make_layers(self, block, in_channels, out_channels, stride, n):

                if stride == 2:
                        down_sample = True
                else: down_sample = False

                layers = nn.ModuleList([block(in_channels, out_channels, stride = stride, down_sample = down_sample)])

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
                out = self.layers4(out)

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

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.down_sample is not None:
                        x = self.down_sample(x)

                out = out + x 
                out = self.relu(out)

                return out 

class IdentityPadding(nn.Module):

        def __init__(self, in_channels, out_channels, stride):
                super(IdentityPadding, self).__init__()

                self.down = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                                nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):

                out = self.down(x)
                return out 

def resnet34():
        return ResNet34([3, 4, 6, 3], block = ResidualBlock)