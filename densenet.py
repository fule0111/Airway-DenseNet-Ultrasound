import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([x, out], 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat([x, out], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate=12, depth=100, reduction=0.5, nClasses=2, bottleneck=True):
        """
        DenseNet constructor.

        Args:
            growthRate (int): growth rate (k)
            depth (int): total depth (number of layers including convs). common choices: 100, 121, etc.
            reduction (float): compression factor for transition layers (0 < reduction <= 1)
            nClasses (int): number of output classes
            bottleneck (bool): whether to use bottleneck layers (DenseNet-BC when True)
        """
        super(DenseNet, self).__init__()

        # compute number of layers in each dense block
        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2  # when using bottleneck, effective number of layers per block halves

        # initial number of channels after first conv
        nChannels = 2 * growthRate

        # initial convolution for single-channel (grayscale) input
        # use kernel_size=7 and stride=2 like many DenseNet variants, followed by maxpool
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # first dense block + transition
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        # second dense block + transition
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        # third dense block (no transition after final block)
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate

        # final batchnorm + classifier
        self.bn_final = nn.BatchNorm2d(nChannels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nChannels, nClasses)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_dense(self, in_channels, growth_rate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(in_channels, growth_rate))
            else:
                layers.append(SingleLayer(in_channels, growth_rate))
            in_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, H, W)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)

        out = self.bn_final(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        logits = self.fc(out)

        # return logits (not softmax/log_softmax) — compatible with nn.CrossEntropyLoss
        return logits
