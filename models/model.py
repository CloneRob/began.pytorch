from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

def weights_init(model):
    """
    custom weight initialization called on netG and netD
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        model.weight.data.normal_(1.0, 0.02)

class Discriminator(nn.Module):
    def __init__(self, ngpu, ngf, ndf, nc, nz):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(ndf, ndf, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf, ndf, 3, 1, 1),
            nn.ReLU(inplace=True),

            # Subsampling
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1),
            nn.ReLU(inplace=True),

            # Subsampling
            nn.Conv2d(ndf * 2, ndf * 3, 3, 2, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(ndf * 3, ndf * 3, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf * 3, ndf * 3, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),

            # Subsampling
            nn.Conv2d(ndf * 3, ndf * 4, 3, 2, 1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
        )
        self.fc_down = nn.Linear(8 * 8 * ndf * 4, nz)
        self.fc_up = nn.Linear(nz, 8 * 8 * ndf)

        self.decoder = nn.Sequential(
            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
            nn.ELU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
            nn.ELU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
            nn.ELU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf, nc, 3, 1, 1, bias=False),
        )

    def forward(self, input):
        # gpu_ids = None
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        # gpu_ids = range(self.ngpu)
        # out = nn.parallel.data_parallel(self.encoder, input, gpu_ids)
        out = self.encoder(input)
        out = self.fc_down(out.view(out.size(0), -1))
        out = self.fc_up(out)
        out = out.view(out.size(0), self.ndf, 8, 8)
        out = self.decoder(out)
        # out = nn.parallel.data_parallel(self.decoder, out, gpu_ids)

        # print('Encoder size after fc: ', out.size())
        return out

class Generator(nn.Module):
    def __init__(self, ngpu, ngf, nc, nz):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf

        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nz, 8 * 8 * ngf)
        self.main = nn.Sequential(
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.ELU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.ELU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.ELU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ngf, nc, 3, 1, 1, bias=False),
        )

    def forward(self, input):
        # gpu_ids = None
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            # gpu_ids = range(self.ngpu)
        x = self.fc(input)
        x = x.view(x.size(0), self.ngf, 8, 8)
        out = self.main(x)
        # out = nn.parallel.data_parallel(self.main, x, gpu_ids)
        return out
