from __future__ import print_function
import math
from enum import Enum
# import torch
import torch.nn as nn
# import torch.nn.parallel

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv_transpose3x3(in_planes, out_planes, stride=2):
    "3x3 convolution with padding"
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)

class BlockType(Enum):
    UP=1
    PLANE=2
    DOWN=3

class ResNetGenerator(nn.Module):
    def __init__(self, layers, ngpu, ngf, nz):
        self.inplanes = nz
        self.ngpu = ngpu
        super().__init__()

        self.decoder = nn.Sequential(
            self._up_block(ngf * 6),
            self._make_layer(BlockType.UP, ngf * 6, layers[3]),
            self._make_layer(BlockType.UP, ngf * 4, layers[2]),
            self._make_layer(BlockType.UP, ngf * 2, layers[1]),
            self._make_layer(BlockType.UP, ngf, layers[0]),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def _up_block(self, planes):
        upconv = nn.ConvTranspose2d(self.inplanes, planes, 4, 1, 0)
        self.inplanes = planes
        return nn.Sequential(upconv)

    def _make_layer(self, block_type, planes, blocks):
        layers = []

        if block_type == BlockType.DOWN:
            layers.append(DownBlock(self.inplanes, planes))
            for _ in range(1, blocks):
                layers.append(PlaneBlock(planes))
            self.inplanes = planes
        elif block_type == BlockType.UP:
            for _ in range(1, blocks):
                layers.append(PlaneBlock(self.inplanes))
            layers.append(UpBlock(self.inplanes, planes))
            self.inplanes = planes
        elif block_type == BlockType.PLANE:
            for _ in range(0, blocks):
                layers.append(PlaneBlock(self.inplanes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.decoder(x)
        return out

class ResNetDiscriminator(nn.Module):
    def __init__(self, layers, ngpu, ndf, nz):
        self.inplanes = ndf
        self.ngpu = ngpu
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=7, stride=2, padding=3, bias=False),
            self._make_layer(BlockType.DOWN, ndf, layers[0]),
            self._make_layer(BlockType.DOWN, ndf * 2, layers[1]),
            self._make_layer(BlockType.DOWN, ndf * 4, layers[2]),
            self._make_layer(BlockType.DOWN, ndf * 6, layers[3]),
            self._make_layer(BlockType.DOWN, nz, layers[4]),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            self._up_block(ndf * 6),
            self._make_layer(BlockType.UP, ndf * 6, layers[3]),
            self._make_layer(BlockType.UP, ndf * 4, layers[2]),
            self._make_layer(BlockType.UP, ndf * 2, layers[1]),
            self._make_layer(BlockType.UP, ndf, layers[0]),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def _up_block(self, planes):
        upconv = nn.ConvTranspose2d(self.inplanes, planes, 4, 1, 0)
        self.inplanes = planes
        return nn.Sequential(upconv)

    def _make_layer(self, block_type, planes, blocks):
        layers = []

        if block_type == BlockType.DOWN:
            layers.append(DownBlock(self.inplanes, planes))
            for _ in range(1, blocks):
                layers.append(PlaneBlock(planes))
            self.inplanes = planes
        elif block_type == BlockType.UP:
            for _ in range(1, blocks):
                layers.append(PlaneBlock(self.inplanes))
            layers.append(UpBlock(self.inplanes, planes))
            self.inplanes = planes
        elif block_type == BlockType.PLANE:
            for _ in range(0, blocks):
                layers.append(PlaneBlock(self.inplanes))

        return nn.Sequential(*layers)

    def forward(self, x):
        code = self.encoder(x)
        out = self.decoder(code)
        return out


class BasicBlock(nn.Module):
    def __init__(self, planes):
        super().__init__()

        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv3x3(planes, planes)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        return out

class PlaneBlock(BasicBlock):
    _type = BlockType.PLANE
    def __init__(self, planes):
        super().__init__(planes)

        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv3x3(planes, planes)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        return residual + BasicBlock.forward(self, out)

class UpBlock(BasicBlock):
    _type = BlockType.UP
    def __init__(self, inplanes, planes):
        super().__init__(inplanes)

        self.up_bn = nn.BatchNorm2d(inplanes)
        self.up_conv = nn.ConvTranspose2d(inplanes, planes, 2, 2, 0, bias=False)
        self.shortcut = nn.ConvTranspose2d(inplanes, planes, 2, 2, 0, bias=False)

    def forward(self, x):
        residual = self.shortcut(x)
        out = BasicBlock.forward(self, x)
        out = self.up_bn(out)
        out = self.relu(out)
        out = self.up_conv(out)
        return residual + out

class DownBlock(BasicBlock):
    _type = BlockType.DOWN
    def __init__(self, inplanes, planes):
        super().__init__(planes)

        self.down_bn = nn.BatchNorm2d(inplanes)
        self.down_conv = conv3x3(inplanes, planes, stride=2)
        self.shortcut = conv1x1(inplanes, planes, stride=2)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.down_bn(x)
        out = self.relu(out)
        out = self.down_conv(out)
        return residual + BasicBlock.forward(self, out)



def resnet18(ngpu, ngf, ndf, nz):
    """Constructs a ResNet-18 based generator and discriminator model.
    """
    disc = ResNetDiscriminator([2, 2, 2, 2, 2], ngpu, ndf, nz)
    gen = ResNetGenerator([2, 2, 2, 2], ngpu, ngf, nz)
    return disc, gen
