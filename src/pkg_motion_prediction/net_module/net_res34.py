import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_

try:
    from submodules import compact_conv_layer as conv
    from net_module.module_wta import *
except:
    from net_module.submodules import compact_conv_layer as conv
    from net_module.module_wta import *

def make_layer(block, in_ch, out_ch, num_blocks, stride=1): # layer 1,2,3,4
    downsample = None
    if stride != 1 or in_ch != out_ch:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        )
    layers = []
    layers.append(block(in_ch, out_ch, stride, downsample))
    for _ in range(num_blocks-1):
        layers.append(block(out_ch, out_ch))
    return nn.Sequential(*layers)

class StemBlock(nn.Module): # deep stem
    def __init__(self, in_channel, deep_stem=False, with_batch_norm=True):
        super().__init__()
        self.deep = deep_stem
        self.stem_channels=[32,32,64]
        if deep_stem:
            self.conv1 = conv(with_batch_norm, in_channel, self.stem_channels[0], kernel_size=3, stride=2, padding=1)
            self.conv2 = conv(with_batch_norm, self.stem_channels[0], self.stem_channels[1], kernel_size=3, stride=1, padding=1)
            self.conv3 = conv(with_batch_norm, self.stem_channels[1], self.stem_channels[2], kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = conv(with_batch_norm, in_channel, self.stem_channels[-1], kernel_size=7, stride=2, padding=3)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.deep:
            out = self.conv3(self.conv2(self.conv1(x)))
        else:
            out = self.conv1(x)
        out = self.pooling(out)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, with_batch_norm=True):
        super().__init__()
        self.conv1 = conv(with_batch_norm, in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.conv2 = conv(with_batch_norm, out_channel, out_channel, kernel_size=3, stride=1, padding=1, activate=False)
        self.downsample = downsample
        self.leaky = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        out += identity
        out = self.leaky(out)
        return out

class ResNet34(nn.Module):
    def __init__(self, in_channel, block,  dim_out, with_batch_norm):
        super().__init__()
        self.dim_out = dim_out
        num_layers = [3,4,6,3]
        num_channels = [64, 128, 256, 512]
        self.stem = StemBlock(in_channel, deep_stem=False, with_batch_norm=with_batch_norm)
        self.layer1 = make_layer(block, in_ch=self.stem.stem_channels[-1], out_ch=num_channels[0], num_blocks=num_layers[0])
        self.layer2 = make_layer(block, in_ch=num_channels[0], out_ch=num_channels[1], num_blocks=num_layers[1], stride=2)
        self.layer3 = make_layer(block, in_ch=num_channels[1], out_ch=num_channels[2], num_blocks=num_layers[2], stride=2)
        self.layer4 = make_layer(block, in_ch=num_channels[2], out_ch=num_channels[3], num_blocks=num_layers[3], stride=2)

        self.apool = nn.AdaptiveAvgPool2d((1,1))
        # self.apool = nn.AvgPool2d(2, 2)
        if dim_out != 0:
            self.fc = nn.Linear(num_channels[3], dim_out)
            # self.fc = nn.Linear(12800, dim_out)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.apool(x)
        x = torch.flatten(x,1)

        if self.dim_out != 0:
            x = self.fc(x)

        return x

class ConvSwarm(nn.Module):
    # batch x channel x height x width
    def __init__(self, input_channel, dim_output, fc_input, num_components, with_batch_norm=True, axes=None):
        super(ConvSwarm,self).__init__()

        self.resnet34  = ResNet34(input_channel, BasicBlock, fc_input, with_batch_norm)

        self.M = num_components
        self.swarm = SwarmModule(fc_input, dim_output, num_components)

        self.axes = axes

    def forward(self, x):
        x = self.resnet34(x)
        x = self.swarm(x)
        return x


if __name__ == '__main__':
    sample_image = torch.rand((2,3,300,300)) # batch x channel x width x height
    model = ConvSwarm(input_channel=3, dim_output=2, fc_input=128, num_components=5)
    sample_out = model(sample_image)
    print(sample_out.shape)
