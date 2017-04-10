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

class ResNetBase(nn.Module):
    """Resnet Base Class, contains methods for
       weight initialization and layer creation
    """
    def __init__(self, ngpu, inplanes):
        self.ngpu = ngpu
        self.inplanes = inplanes
        super().__init__()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        pass

class ResNetEncoder(ResNetBase):
    """ResNet encoder mapping the input to dim(nz)
    """
    def __init__(self, layers, ngpu, ndf, nz):
        super().__init__(ngpu, inplanes=ndf)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=7, stride=2, padding=3, bias=False),
            self._make_layer(BlockType.DOWN, ndf, layers[0]),
            self._make_layer(BlockType.DOWN, ndf * 2, layers[1]),
            self._make_layer(BlockType.DOWN, ndf * 4, layers[2]),
            self._make_layer(BlockType.DOWN, ndf * 6, layers[3]),
            self._make_layer(BlockType.DOWN, nz, layers[4]),
            nn.Tanh()
        )
        self._weight_init()

    def forward(self, x):
        out = self.encoder(x)
        return out

class ResNetGenerator(ResNetBase):
    """mapping the input dim(nz) to an image
    """
    def __init__(self, layers, ngpu, ngf, nz):
        super().__init__(ngpu, inplanes=nz)

        self.decoder = nn.Sequential(
            self._up_block(ngf * 6),
            self._make_layer(BlockType.UP, ngf * 6, layers[3]),
            self._make_layer(BlockType.UP, ngf * 4, layers[2]),
            self._make_layer(BlockType.UP, ngf * 2, layers[1]),
            self._make_layer(BlockType.UP, ngf, layers[0]),
        )
        self.transformer = nn.Sequential(
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )
        self._weight_init()

    def forward(self, x):
        out = self.transformer(self.decoder(x))
        return out

    def features(self, x):
        """Extract generator features
        """
        return self.decoder(x)

class ResNetDiscriminator(ResNetBase):
    """ResNet autoencoder
    """
    def __init__(self, layers, ngpu, ngf, ndf, nz):
        super().__init__(ngpu, inplanes=ndf)
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
            self._up_block(ngf * 6),
            self._make_layer(BlockType.UP, ngf * 6, layers[3]),
            self._make_layer(BlockType.UP, ngf * 4, layers[2]),
            self._make_layer(BlockType.UP, ngf * 2, layers[1]),
            self._make_layer(BlockType.UP, ngf, layers[0]),
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        self._weight_init()

    def forward(self, x):
        features = self.decoder(self.encoder(x))
        return self.classifier(features)

    def features(self, x):
        """Extraxt the penultimate features of the discriminator
        """
        return self.decoder(self.encoder(x))



class BasicBlock(nn.Module):
    """Basic ResNet building block
    """
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
    """ResNet block keeping input resolution and dimension unchanged
    """
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
    """ResNet upsampling block
    """
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
    """ResNet downsampling block
    """
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
    disc = ResNetDiscriminator([1, 1, 1, 1, 1], ngpu, ngf, ndf, nz)
    gen = ResNetGenerator([1, 1, 1, 1], ngpu, ngf, nz)
    return disc, gen

