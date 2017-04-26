"""
Implementation of Boundary Equilibrium GAN, arXiv:1703.10717v1
"""
from __future__ import print_function
import os
import random

from datetime import datetime

import util
import config
from models import model, dcgan
from models.resnet import resnet18, resnet34, resnet
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


opt = config.get_config()
print(opt)

# if opt.device_id != '':
#     os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_id

try:
    os.makedirs(os.path.join(opt.outf, 'samples'))
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


def began(generator, discriminator, dataloader, criterion, aux):
    """Training of gan network
    """
    feedback_control = Variable(
        torch.cuda.FloatTensor([0.001]), requires_grad=False)
    gamma = Variable(torch.cuda.FloatTensor([0.4]), requires_grad=False)
    k_var = Variable(torch.cuda.FloatTensor(
        [float(opt.klr)]), requires_grad=False)

    optimizerD = optim.Adam(discriminator.parameters(),
                            lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(),
                            lr=opt.lr, betas=(opt.beta1, 0.999))

    for epoch in range(opt.start_epoch, opt.niter):
        for i, (data, _) in enumerate(dataloader, 0):

            data = data.cuda()
            data_var = Variable(data)
            batch_size = data_var.size(0)

            discriminator.zero_grad()
            generator.zero_grad()

            real_reconstruction = discriminator(data_var)
            discriminator_realloss = criterion(real_reconstruction, data_var)

            aux.noise.data.resize_(batch_size, aux.nz, 1, 1)
            aux.noise.data.uniform_(-1, 1)
            noise_d_sample = generator(aux.noise)
            noise_reconstruction = discriminator(noise_d_sample)
            # discriminator_genloss = criterion(
            #     noise_reconstruction, noise_d_sample)
            discriminator_genloss = torch.mean(torch.abs(noise_reconstruction - noise_d_sample))

            err_discriminator = discriminator_realloss - k_var * discriminator_genloss
            err_discriminator.backward(retain_variables=True)
            optimizerD.step()
            discriminator_genloss.backward()
            optimizerG.step()

            equilibrium = (gamma * discriminator_realloss -
                           discriminator_genloss).detach()
            k_var += feedback_control * equilibrium
            k_var.data.clamp_(opt.lb, 1.0)
            global_measure = discriminator_realloss + equilibrium.norm()

            if i % 10 == 0:
                format_str = ('{}[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f}, D(x): {:.4f}, D(G(z)): {:.4f}'
                              ', Global: {:.4f}, k: {:.4f}')
                print(format_str.format(
                    datetime.now().time(),
                    epoch, opt.niter, i, len(dataloader),
                    err_discriminator.data[0],
                    discriminator_genloss.data[0],
                    discriminator_realloss.data[0],
                    discriminator_genloss.data[0],
                    global_measure.data[0], k_var.data[0]))

            if i % 125 == 0:
                vutils.save_image(data_var.data.add(aux.mean.expand_as(data_var)),
                                  '%s/samples/%03d-%03d_real_samples.png' % (opt.outf, i, epoch))
                vutils.save_image(real_reconstruction.data.add(aux.mean.expand_as(data_var)),
                                  '%s/samples/%03d-%03d_real_reconstruction.png' % (opt.outf, i, epoch))

                fake = generator(aux.noise)
                vutils.save_image(fake.data.add(aux.mean.expand_as(data_var)),
                                  '%s/samples/%03d-%03d_fake_samples.png' % (opt.outf, i, epoch))
        torch.save(generator.state_dict(), '%s/generator%d.pth' % (opt.outf, epoch))
        torch.save(discriminator.state_dict(), '%s/discriminator%d.pth' % (opt.outf, epoch))


def get_dataloader():
    """
    Creates the dataloader from the given imagepath
    Barrett Data Mean: R: 0.5039812792256271, G: 0.40989934259960137, B: 0.3683006199917145
    Barrett Data Std: R: 0.01237758766377634, G: 0.011287555487577814, B: 0.010316500128800232
    """
    print('Hi from dataloader')
    root = os.path.join(opt.dataroot, opt.dataset)
    dataset = dset.ImageFolder(root=root,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize + 8),
                                   local_transforms.RandomRotation(),
                                   transforms.RandomCrop(opt.imageSize),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.50398127, 0.40989934, 0.36830061),
                                                       std=(1.0, 1.0, 1.0)),
                               ]))

    assert dataset
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
        fixed_noise = torch.FloatTensor(
            batch_size, latent_dim, 1, 1).uniform_(-1, 1)
        label = torch.FloatTensor(batch_size)

        mean_tensor = torch.FloatTensor(3, opt.imageSize, opt.imageSize)
        mean_tensor[0].fill_(0.50398127)
        mean_tensor[1].fill_(0.40989934)
        mean_tensor[2].fill_(0.36830061)
        std_tensor = torch.FloatTensor(3, opt.imageSize, opt.imageSize)
        std_tensor[0].fill_(0.0123775)
        std_tensor[1].fill_(0.01128755)
        std_tensor[2].fill_(0.01031650)

        if opt.cuda:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            mean_tensor, std_tensor = mean_tensor.cuda(), std_tensor.cuda()
            label = label.cuda()

        self.noise = Variable(noise, requires_grad=False)
        self.fixed_noise = Variable(fixed_noise, requires_grad=False)
        self.label = Variable(label)
        self.nz = latent_dim
        self.mean = mean_tensor
        self.std = std_tensor


def load_model(nc=3):
    """
    Creates and initializes the generator and
    discriminator model
    """
    print('Hi from loadmodel')
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    ngpu = int(opt.ngpu)

    discriminator = dcgan.DiscriminatorUp(ngpu, ngf, ndf, nc, nz)
    # discriminator = model.Discriminator(ngpu, ndf, nc, nz)
    discriminator.apply(model.weights_init)
    print(discriminator)
    # generator = model.Generator(ngpu, ngf, nc, nz)
    generator = dcgan.GeneratorUp(ngpu, ngf, nc, nz)
    generator.apply(model.weights_init)
    print(generator)

    if opt.cuda:
        discriminator.cuda()
        generator.cuda()
        # discriminator = nn.DataParallel(discriminator)
        # generator = nn.DataParallel(generator)

    return generator, discriminator


if __name__ == '__main__':
    main()
    # test()
