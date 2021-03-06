from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

class GeneratorUp(nn.Module):
    def __init__(self, ngpu, ngf, nc, nz):
        super(GeneratorUp, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf*8) x 4 x 4
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 8),
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 4),
            # state size. (ngf*4) x 8 x 8
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 4),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 2),
            # state size. (ngf*2) x 16 x 16
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf),
            # state size. (ngf) x 32 x 32
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf),
            nn.Conv2d(ngf, nc, 3, 1, 1, bias=False),
            # state size. (nc) x 64 x 64
            nn.Tanh()
        )

    def forward(self, input):
        """
        gpu_ids = [0]
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu >= 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)
        """
        return self.main(input)

class Generator(nn.Module):
    def __init__(self, ngpu, ngf, nc, nz):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 4),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 2),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf),
            nn.Conv2d(ngf, nc, 3, 1, 1),
            # state size. (nc) x 64 x 64
            nn.Tanh(),
        )

    def forward(self, input):
        """
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu >= 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)
        """
        return self.main(input)

#####################################################################
class DiscriminatorUp(nn.Module):
    def __init__(self, ngpu, ngf, ndf, nc, nz):
        super(DiscriminatorUp, self).__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf * 2),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf * 4),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf * 8),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, nz, 4, 1, 0, bias=False),
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf*8) x 4 x 4
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 8),
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 4),
            # state size. (ngf*4) x 8 x 8
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 4),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 2),
            # state size. (ngf*2) x 16 x 16
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf),
            # state size. (ngf) x 32 x 32
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf),
            nn.Conv2d(ngf, nc, 3, 1, 1, bias=False),
            # state size. (nc) x 64 x 64
            nn.Tanh()
        )

    def forward(self, input):
        """
        gpu_ids = [0]
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.encoder, input, gpu_ids)
        """
        code = self.encoder(input)
        torch.clamp(code, -1.0, 1.0)
        return self.decoder(code)

class Discriminator(nn.Module):
    def __init__(self, ngpu, ngf, ndf, nc, nz):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf * 2),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf * 4),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf * 8),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, nz, 4, 1, 0),
            nn.Tanh(),


            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 8),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 4),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 2),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf),
            nn.Conv2d(ngf, nc, 3, 1, 1),
        )

    def forward(self, input):
        """
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
