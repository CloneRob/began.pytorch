from __future__ import print_function
import math
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

class Discriminator(nn.Module):
    def __init__(self, ngpu, ngf, ndf, nc, nz):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, nz, 4, 1, 0, bias=False),
            nn.Tanh(),

            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, nc, 3, 1, 1, bias=False),
        )

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        output = nn.parallel.data_parallel(self.encoder, input, gpu_ids)
        # print(code.size())
        # output = nn.parallel.data_parallel(self.decoder, code, gpu_ids)
        # print(output.size())
        return output
        """
        return self.encoder(input)
        """

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv_transpose3x3(in_planes, out_planes, stride=2):
    "3x3 convolution with padding"
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)


class ResNetDiscriminator(nn.Module):

    def __init__(self, block_down, block_up, layers, ndf, nz):
        self.inplanes = ndf
        super(ResNetDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, ndf, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.down_layer1 = self._make_layer(block_down, ndf, layers[0], stride=2)
        self.down_layer2 = self._make_layer(block_down, ndf * 2, layers[1], stride=2)
        self.down_layer3 = self._make_layer(block_down, ndf * 3, layers[2], stride=2)
        self.down_layer4 = self._make_layer(block_down, ndf * 4, layers[3], stride=2)
        self.down_layer5 = self._make_layer(block_down, nz, layers[4], stride=2)
        print('test')
        self.tanh = nn.Tanh()

        self.up_layer1 = self._make_layer(block_up, ndf * 4, layers[3], stride=2)
        self.up_layer2 = self._make_layer(block_up, ndf * 3, layers[2], stride=2)
        self.up_layer3 = self._make_layer(block_up, ndf * 3, layers[2], stride=2)
        self.up_layer4 = self._make_layer(block_up, ndf * 2, layers[1], stride=2)
        self.up_layer5 = self._make_layer(block_up, ndf * 2, layers[1], stride=2)
        self.up_layer6 = self._make_layer(block_up, ndf, layers[0], stride=2)

        self.bn = nn.BatchNorm2d(ndf)
        self.conv2 = nn.Conv2d(ndf, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        sampler = None
        if stride != 1 or self.inplanes != planes * block.expansions:
            if block._type() == BlockType.DOWN:
                sampler = DownSample(self.inplanes, planes * block.expansion)
                layers.append(block(self.inplanes, planes, stride, sampler))
                self.inplanes = planes * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, planes))

            elif block._type() == BlockType.UP:
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, self.inplanes))
                sampler = UpSample(self.inplanes, planes * block.expansion)
                layers.append(block(self.inplanes, planes, stride, sampler))
                self.inplanes = planes

        else:
            self.inplanes = planes * block.expansion
            for _ in range(0, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.down_layer1(x)
        x = self.down_layer2(x)
        x = self.down_layer3(x)
        x = self.down_layer4(x)
        x = self.down_layer5(x)
        x = self.tanh(x)

        x = self.up_layer1(x)
        x = self.up_layer2(x)
        x = self.up_layer3(x)
        x = self.up_layer4(x)
        x = self.up_layer5(x)
        x = self.up_layer6(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.tanh(x)

        return x

class BlockType(Enum):
    UP = 1
    DOWN = 2

class BasicUp(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(BasicUp, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.conv2 = conv3x3(inplanes, planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        if self.upsample is not None:
            residual = self.upsample(x)
            out = self.up(out)
        out = self.conv2(out)


        # print('out size: {}, residual size: {}'.format(out.size(), residual.size()))
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
        self.up = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        return self.up(x)

class BasicDown(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicDown, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

    @staticmethod
    def sampler(inplanes, planes):
        return DownSample(inplanes, planes)

    @staticmethod
    def _type():
        return BlockType.DOWN

class DownSample(nn.Module):
    def __init__(self, inplanes, planes):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, 2, 0, bias=False),
            nn.BatchNorm2d(planes)
        )
    def forward(self, x):
        return self.down(x)



def resnet18(ndf, nz):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetDiscriminator(BasicDown, BasicUp, [1, 1, 1, 1, 1], ndf, nz)
    return model
