import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_

from motion_prediction.net_module.submodules import compact_conv_layer as conv
from motion_prediction.net_module.module_wta import *

from motion_prediction.net_module import module_mdn

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

class ResNet34Lite(nn.Module):
    def __init__(self, in_channel, block, with_batch_norm):
        super().__init__()
        num_layers = [3,4,6,3]
        num_channels = [16, 32, 64, 128]
        self.stem = StemBlock(in_channel, deep_stem=False, with_batch_norm=with_batch_norm)
        self.layer1 = make_layer(block, in_ch=self.stem.stem_channels[-1], out_ch=num_channels[0], num_blocks=num_layers[0])
        self.layer2 = make_layer(block, in_ch=num_channels[0], out_ch=num_channels[1], num_blocks=num_layers[1], stride=2)
        self.layer3 = make_layer(block, in_ch=num_channels[1], out_ch=num_channels[2], num_blocks=num_layers[2], stride=2)
        self.layer4 = make_layer(block, in_ch=num_channels[2], out_ch=num_channels[3], num_blocks=num_layers[3], stride=2)
        self.apool  = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.apool(x)
        return x

class ResNet34(nn.Module):
    def __init__(self, in_channel, block, with_batch_norm):
        super().__init__()
        num_layers = [3,4,6,3]
        num_channels = [64, 128, 256, 512]
        self.stem = StemBlock(in_channel, deep_stem=True, with_batch_norm=with_batch_norm)
        self.layer1 = make_layer(block, in_ch=self.stem.stem_channels[-1], out_ch=num_channels[0], num_blocks=num_layers[0])
        self.layer2 = make_layer(block, in_ch=num_channels[0], out_ch=num_channels[1], num_blocks=num_layers[1], stride=2)
        self.layer3 = make_layer(block, in_ch=num_channels[1], out_ch=num_channels[2], num_blocks=num_layers[2], stride=2)
        self.layer4 = make_layer(block, in_ch=num_channels[2], out_ch=num_channels[3], num_blocks=num_layers[3], stride=2)
        self.apool  = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.apool(x)
        return x


class ConvMultiHypoNet(nn.Module):
    # batch x channel x height x width
    def __init__(self, input_channel, dim_output, fc_input, num_components, with_batch_norm=True, axes=None, lite=True):
        super(ConvMultiHypoNet,self).__init__()

        if lite:
            self.resnet34 = ResNet34Lite(input_channel, BasicBlock, with_batch_norm)
        else:
            self.resnet34 = ResNet34(input_channel, BasicBlock, with_batch_norm)

        if lite:
            self.fc1   = nn.Linear(fc_input,128)
        else:
            self.fc1   = nn.Linear(fc_input,1024)
        self.leaky = nn.LeakyReLU(inplace=True)

        self.M = num_components
        if lite:
            self.swarm = MultiHypothesisModule(128, dim_output, num_components)
        else:
            self.swarm = MultiHypothesisModule(1024, dim_output, num_components)
        # self.swarm = AdaptiveSwarmModule(128, dim_output, num_components)

        self.axes = axes

    def forward(self, x):
        out_conv = self.resnet34(x)

        if self.axes is not None:
            for i, ax in enumerate(self.axes.ravel()):
                ax.cla()
                ax.imshow(out_conv[0,i,:,:].cpu().detach().numpy())

        x = out_conv.view(out_conv.size(0), -1) # batch x -1
        x = self.leaky(self.fc1(x))
        x = self.swarm(x)

        return x

class ConvMixtureDensityNet(nn.Module):
    # batch x channel x height x width
    def __init__(self, input_channel, dim_output, fc_input, num_components, with_batch_norm=True, axes=None):
        super(ConvMixtureDensityNet,self).__init__()

        self.resnet34 = ResNet34Lite(input_channel, BasicBlock, with_batch_norm)

        self.fc1   = nn.Linear(fc_input,128)
        self.leaky = nn.LeakyReLU(inplace=True)

        self.M = num_components
        self.mdn = module_mdn.ClassicMixtureDensityModule(128, dim_output, num_components)

        self.axes = axes

    def forward(self, x):
        out_conv = self.resnet34(x)

        if self.axes is not None:
            for i, ax in enumerate(self.axes.ravel()):
                ax.cla()
                ax.imshow(out_conv[0,i,:,:].cpu().detach().numpy())

        x = out_conv.view(out_conv.size(0), -1) # batch x -1
        x = self.leaky(self.fc1(x))
        x = self.mdn(x)

        return x

class ConvMixtureDensityFit(nn.Module):
    # batch x channel x height x width
    def __init__(self, multi_hypo_net, dim_output, num_hypos, num_gaus, with_batch_norm=True, axes=None):
        super(ConvMixtureDensityFit,self).__init__()

        self.multihyponet = multi_hypo_net
        for param in self.multihyponet.parameters():
            param.requires_grad = False
        
        self.M = num_gaus
        self.smdn = module_mdn.SamplingMixtureDensityModule(dim_output, num_hypos, num_gaus)

        self.axes = axes

    def forward(self, x, device='cuda'):
        x = self.multihyponet(x)
        x = self.smdn(x, device=device)
        return x


class ConvMultiHypoMixtureDensityFit(nn.Module):
    # batch x channel x height x width
    def __init__(self, input_channel, dim_output, fc_input, num_hypos, num_gaus, with_batch_norm=True, axes=None):
        super(ConvMultiHypoMixtureDensityFit,self).__init__()

        self.resnet34 = ResNet34Lite(input_channel, BasicBlock, with_batch_norm)

        self.fc1   = nn.Linear(fc_input,128)
        self.leaky = nn.LeakyReLU(inplace=True)

        self.K = num_hypos
        self.M = num_gaus
        self.swarm = MultiHypothesisModule(128, dim_output, num_hypos)
        self.smdn = module_mdn.SamplingMixtureDensityModule(dim_output, num_hypos, num_gaus)

        self.axes = axes

    def forward(self, x, device='cuda'):
        out_conv = self.resnet34(x)

        if self.axes is not None:
            for i, ax in enumerate(self.axes.ravel()):
                ax.cla()
                ax.imshow(out_conv[0,i,:,:].cpu().detach().numpy())

        x = out_conv.view(out_conv.size(0), -1) # batch x -1
        x = self.leaky(self.fc1(x))
        x = self.swarm(x)

        x = self.smdn(x, device=device)

        return x

