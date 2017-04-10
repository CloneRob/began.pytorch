"""
Implementation of Boundary Equilibrium GAN, arXiv:1703.10717v1
"""
from __future__ import print_function
import argparse
import os
import random

from models import model, dcgan
from models.resnet import resnet18
import transformer as local_transforms

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


# pylint: disable=C0103

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, help='Location of data, combined with dataroot',
                    default='Augsburg/barrett_split/circular_split/128x128')
parser.add_argument('--dataroot', required=False, help='Dataset root', default='/mnt/datassd/Projects/Datasets/')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate, default=0.0002')
parser.add_argument('--klr', type=float, default=0.001, help='learning rate for k, default=0.0002')
parser.add_argument('--lb', type=float, default=0.0, help='lower bound for k, default=0.0')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gen', default='', help="path to netG (to continue training)")
parser.add_argument('--dis', default='', help="path to netD (to continue training)")
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--outf', default='./samples/sample0', help='folder to output images and model checkpoints')
parser.add_argument('--device_id', default='', help='GPU device id where the model is run')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, default=13)




opt = parser.parse_args()
print(opt)

if opt.device_id != '':
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_id

try:
    os.makedirs(opt.outf)
except OSError:
    pass
print("Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

def main():
    """
    main function
    """
    dataloader = get_dataloader()
    nz = int(opt.nz)

    generator, discriminator = load_model()

    criterion = nn.L1Loss().cuda()

    auxillary_variables = AuxilaryVariables(opt.batch_size, nz)

    began(generator, discriminator, dataloader, criterion, auxillary_variables)
    # sample(generator, discriminator, auxillary_variables)


def began(generator, discriminator, dataloader, criterion, aux):
    """Training of gan network
    """
    feedback_control = Variable(torch.cuda.FloatTensor([0.01]), requires_grad=False)
    gamma = Variable(torch.cuda.FloatTensor([0.6]), requires_grad=False)
    k_var = Variable(torch.cuda.FloatTensor([float(opt.klr)]), requires_grad=False)

    optimizerD = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    for epoch in range(opt.start_epoch, opt.niter):
        for i, (data, _) in enumerate(dataloader, 0):
            discriminator.zero_grad()

            data = data.cuda()
            data_var = Variable(data)
            batch_size = data_var.size(0)

            optimizerD.zero_grad()
            for p in discriminator.parameters():
                p.requires_grad = True
            for p in generator.parameters():
                p.requires_grad = False


            real_reconstruction = discriminator(data_var)
            discriminator_realloss = criterion(real_reconstruction, data_var)

            aux.noise.data.resize_(batch_size, aux.nz, 1, 1)
            aux.noise.data.uniform_(-1, 1)
            noise_d_sample = generator(aux.noise).detach()
            noise_reconstruction = discriminator(noise_d_sample)
            discriminator_genloss = criterion(noise_reconstruction, noise_d_sample)

            err_discriminator = discriminator_realloss - k_var * discriminator_genloss
            err_discriminator.backward()
            optimizerD.step()

            #################################
            #       Generator training      #
            #################################
            optimizerG.zero_grad()
            for p in discriminator.parameters():
                p.requires_grad = False
            for p in generator.parameters():
                p.requires_grad = True

            generator.zero_grad()
            aux.noise.data.uniform_(-1, 1)

            noise_g_sample = generator(aux.noise)
            discriminator_sample = discriminator(noise_g_sample).detach()

            generator_loss = criterion(noise_g_sample, discriminator_sample)
            generator_loss.backward()
            optimizerG.step()

            equilibrium = (gamma * discriminator_realloss - generator_loss).detach()
            k_var += feedback_control * equilibrium
            k_var.data.clamp_(opt.lb, 1.0)
            global_measure = discriminator_realloss  + equilibrium.norm()

            format_str = ('[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f}, D(x): {:.4f}, D(G(z)): {:.4f}'
                          ', Global: {:.4f}, k: {:.4f}')
            print(format_str.format(
                epoch, opt.niter, i, len(dataloader),
                err_discriminator.data[0],
                generator_loss.data[0],
                discriminator_realloss.data[0],
                discriminator_genloss.data[0],
                global_measure.data[0], k_var.data[0]))

            if i % 125 == 0:
                vutils.save_image(data_var.data,
                                  '%s/%03d_real_samples.png' % (opt.outf, epoch))
                vutils.save_image(real_reconstruction.data,
                                  '%s/%03d_real_reconstruction.png' % (opt.outf, epoch))

                fake = generator(aux.noise)
                vutils.save_image(fake.data,
                                  '%s/%03d_fake_samples.png' % (opt.outf, epoch))

        # do checkpointing
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), '%s/generator%d.pth' % (opt.outf, epoch))
            torch.save(discriminator.state_dict(), '%s/discriminator%d.pth' % (opt.outf, epoch))


def get_dataloader():
    """
    Creates the dataloader from the given imagepath
    """
    root = os.path.join(opt.dataroot, opt.dataset)
    dataset = dset.ImageFolder(root=root,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize + 8),
                                   local_transforms.RandomRotation(),
                                   transforms.RandomCrop(opt.imageSize),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    assert dataset
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=True, num_workers=int(opt.workers),
                                             pin_memory=True)
    return dataloader

class AuxilaryVariables():
    """Class that holds the auxillary variables used
       during training
    """
    def __init__(self, batch_size, latent_dim):
        noise = torch.FloatTensor(batch_size, latent_dim, 1, 1)
        fixed_noise = torch.FloatTensor(batch_size, latent_dim, 1, 1).normal_(0, 1)
        _rlabel = torch.FloatTensor(batch_size).cuda()
        _flabel = torch.FloatTensor(batch_size).cuda()
        _rlabel.fill_(1)
        _flabel.fill_(0)

        if opt.cuda:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        # input = Variable(input)
        # real_label = Variable(label)
        self.real_label = Variable(_rlabel)
        self.fake_label = Variable(_flabel)
        self.noise = Variable(noise)
        self.fixed_noise = Variable(fixed_noise)
        self.nz = latent_dim


def load_model(nc=3):
    """
    Creates and initializes the generator and
    discriminator model
    """
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    ngpu = int(opt.ngpu)

    discriminator, generator = resnet18(ngpu, ngf, ndf, nz)
    # generator.apply(model.weights_init)
    if opt.gen != '':
        generator.load_state_dict(torch.load(opt.gen))
    print(generator)


    # discriminator.apply(model.weights_init)
    if opt.dis != '':
        discriminator.load_state_dict(torch.load(opt.dis))

    if opt.cuda:
        discriminator.cuda()
        generator.cuda()
        discriminator = nn.DataParallel(discriminator)
        generator = nn.DataParallel(generator)

    print(discriminator)
    return generator, discriminator


if __name__ == '__main__':
    main()
    # test()
