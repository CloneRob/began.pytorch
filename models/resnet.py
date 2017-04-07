from __future__ import print_function
import math
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv_transpose3x3(in_planes, out_planes, stride=2):
    "3x3 convolution with padding"
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)


class ResNetDiscriminator(nn.Module):

    def __init__(self, block_down, block_up, layers, ngpu, ndf, nz):
        self.inplanes = ndf
        self.ngpu = ngpu
        super(ResNetDiscriminator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=7, stride=2, padding=3, bias=False),
            self._make_layer(block_down, ndf, layers[0], stride=2),
            self._make_layer(block_down, ndf * 2, layers[1], stride=2),
            self._make_layer(block_down, ndf * 4, layers[2], stride=2),
            self._make_layer(block_down, ndf * 6, layers[3], stride=2),
            self._make_layer(block_down, nz, layers[4], stride=2),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            self._make_layer(block_up, ndf * 6, layers[3], stride=2),
            self._make_layer(block_up, ndf * 4, layers[2], stride=2),
            self._make_layer(block_up, ndf * 4, layers[2], stride=2),
            self._make_layer(block_up, ndf * 2, layers[1], stride=2),
            self._make_layer(block_up, ndf, layers[0], stride=2),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        sampler = None

        if stride != 1:
            if block._type() == BlockType.DOWN:
                sampler = DownSample(self.inplanes, planes * block.expansion, stride).cuda()
                layers.append(block(self.inplanes, planes * block.expansion, sampler))
                self.inplanes = planes * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, planes))

            elif block._type() == BlockType.UP:
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, self.inplanes))
                sampler = UpSample(self.inplanes, planes * block.expansion).cuda()
                layers.append(block(self.inplanes, planes, stride, sampler))
                self.inplanes = planes

        else:
            self.inplanes = planes * block.expansion
            for _ in range(0, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class BlockType(Enum):
    UP = 1
    DOWN = 2

class BasicUp(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(BasicUp, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, inplanes)

        self.bn2 = nn.BatchNorm2d(inplanes)
        self.upsample = upsample

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)

        if self.upsample is not None:
            residual, out = self.upsample(residual, out)

        out += residual
        return out

    @staticmethod
    def sampler(inplanes, planes):
        return UpSample(inplanes, planes)

    @staticmethod
    def _type():
        return BlockType.UP


class UpSample(nn.Module):

    def __init__(self, inplanes, planes):
        super(UpSample, self).__init__()
        self.identity_branch = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False),
        )
        self.weight_branch = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False),
        )

    def forward(self, x1, x2):
        return self.identity_branch(x1), self.weight_branch(x2)

class BasicDown(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None):
        super(BasicDown, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual, out = self.downsample(residual, out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual
        return out

    @staticmethod
    def sampler(inplanes, planes):
        return DownSample(inplanes, planes)

    @staticmethod
    def _type():
        return BlockType.DOWN

class DownSample(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super(DownSample, self).__init__()
        self.identity_branch = nn.Conv2d(inplanes, planes, 1, stride, 0, bias=False)
        self.weight_branch = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)

    def forward(self, x1, x2):
        return self.identity_branch(x1), self.weight_branch(x2)



def resnet18(ngpu, ndf, nz):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetDiscriminator(BasicDown, BasicUp, [1, 1, 1, 1, 1], ngpu, ndf, nz)
    return model
