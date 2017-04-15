"""
Implementation of Boundary Equilibrium GAN, arXiv:1703.10717v1
"""
from __future__ import print_function
import argparse
import os
import random

from datetime import datetime

import util
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
parser.add_argument('--dlr', type=float, default=0.00005, help='learning rate, default=0.0002')
parser.add_argument('--glr', type=float, default=0.00005, help='learning rate, default=0.0002')
parser.add_argument('--klr', type=float, default=0.0, help='learning rate for k, default=0.0002')
parser.add_argument('--lb', type=float, default=0.0, help='lower bound for k, default=0.0')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gen', default='', help="path to netG (to continue training)")
parser.add_argument('--dis', default='', help="path to netD (to continue training)")
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--outf', default='./samples/sample0', help='folder to output images and model checkpoints')
parser.add_argument('--device_id', default='', help='GPU device id where the model is run')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, default=99)




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
    # util.normalization_values(os.path.join(opt.dataroot, 'ImageNet/ILSVRC2012_img_train'), 224)
    # util.normalization_values(os.path.join(opt.dataroot, opt.dataset), 128)
    # return
    dataloader = get_dataloader()
    nz = int(opt.nz)

    generator, discriminator = load_model()

    criterion = nn.L1Loss().cuda()
    # criterion = nn.MSELoss().cuda()

    auxillary_variables = AuxilaryVariables(opt.batch_size, nz)

    began(generator, discriminator, dataloader, criterion, auxillary_variables)

def began2(generator, discriminator, dataloader, criterion, criterion_2, aux):
    """Training of gan network
    """
    feedback_control = Variable(torch.cuda.FloatTensor([0.001]), requires_grad=False)
    gamma = Variable(torch.cuda.FloatTensor([0.4]), requires_grad=False)
    k_var = Variable(torch.cuda.FloatTensor([float(opt.klr)]), requires_grad=False)

    optimizerD = optim.Adam(discriminator.parameters(), lr=opt.dlr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=opt.glr, betas=(opt.beta1, 0.999))

    generator.train()
    discriminator.train()

    for epoch in range(opt.start_epoch, opt.niter):
        util.adjust_lr(optimizerD, epoch, opt.dlr)
        util.adjust_lr(optimizerG, epoch, opt.glr)
        for i, (data, _) in enumerate(dataloader, 0):

            data = data.cuda()
            data_var = Variable(data)
            batch_size = data_var.size(0)

            aux.noise.data.resize_(batch_size, aux.nz, 1, 1)
            aux.noise.data.uniform_(-1, 1)
            aux.label.data.resize_(batch_size).fill_(1)

            for p in discriminator.parameters():
                p.requires_grad = False
            for p in generator.parameters():
                p.requires_grad = True
            generator.zero_grad()

            noise_g_sample = generator(aux.noise)
            discriminator_sample, fake_label = discriminator(noise_g_sample)
            # discriminator_sample.detach()
            discriminator_sample.detach_()
            generator_loss = criterion(noise_g_sample, discriminator_sample)
            gen2_loss = criterion_2(fake_label, aux.label)
            generator_loss += gen2_loss
            generator_loss.backward()


            discriminator.zero_grad()
            for p in discriminator.parameters():
                p.requires_grad = True
            for p in generator.parameters():
                p.requires_grad = False


            real_reconstruction, real_label = discriminator(data_var)
            discriminator_realloss = criterion(real_reconstruction, data_var)

            aux.noise.data.uniform_(-1, 1)
            aux.label.data.fill_(1)
            noise_d_sample = generator(aux.noise).detach()
            noise_reconstruction, fake_label2 = discriminator(noise_d_sample)
            discriminator_genloss = criterion(noise_reconstruction, noise_d_sample)
            real_label_loss = criterion_2(real_label, aux.label)
            aux.label.data.fill_(0)
            fake_label_loss = criterion_2(fake_label2, aux.label)
            err_discriminator = discriminator_realloss + real_label_loss - k_var * (discriminator_genloss + fake_label_loss)
            err_discriminator.backward()


            #############################
            #     Weight updates
            optimizerD.step()
            optimizerG.step()
            #############################

            equilibrium = (gamma * discriminator_realloss - generator_loss).detach()
            k_var += feedback_control * equilibrium
            k_var.data.clamp_(opt.lb, 1.0)

            if i % 10 == 0:
                global_measure = discriminator_realloss  + equilibrium.norm()
                format_str = ('{} | [{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f}, D(x): {:.4f}, D(G(z)): {:.4f}'
                              ', Global: {:.4f}, k: {:.4f}')
                print(format_str.format(
                    datetime.now().time(),
                    epoch, opt.niter, i, len(dataloader),
                    err_discriminator.data[0],
                    generator_loss.data[0],
                    discriminator_realloss.data[0],
                    discriminator_genloss.data[0],
                    global_measure.data[0], k_var.data[0]))

            if i % 125 == 0:
                vutils.save_image(data_var.data, '%s/%03d_real_samples.png' % (opt.outf, epoch))
                vutils.save_image(real_reconstruction.data, '%s/%03d_real_reconstruction.png' % (opt.outf, epoch))

                fake = generator(aux.fixed_noise)
                vutils.save_image(fake.data, '%s/%03d_fake_samples.png' % (opt.outf, epoch))

        # do checkpointing
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), '%s/generator%d.pth' % (opt.outf, epoch))
            torch.save(discriminator.state_dict(), '%s/discriminator%d.pth' % (opt.outf, epoch))

def began(generator, discriminator, dataloader, criterion, aux):
    """Training of gan network
    """
    feedback_control = Variable(torch.cuda.FloatTensor([0.001]), requires_grad=False)
    gamma = Variable(torch.cuda.FloatTensor([0.4]), requires_grad=False)
    k_var = Variable(torch.cuda.FloatTensor([float(opt.klr)]), requires_grad=False)

    optimizerD = optim.Adam(discriminator.parameters(), lr=opt.dlr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=opt.glr, betas=(opt.beta1, 0.999))

    generator.train()
    discriminator.train()

    mone = torch.cuda.FloatTensor([-1])

    for epoch in range(opt.start_epoch, opt.niter):
        util.adjust_lr(optimizerD, epoch, opt.dlr)
        util.adjust_lr(optimizerG, epoch, opt.glr)
        for i, (data, _) in enumerate(dataloader, 0):

            data = data.cuda()
            data_var = Variable(data)
            batch_size = data_var.size(0)

            for p in generator.parameters():
                p.requires_grad = False
            optimizerD.zero_grad()


            real_reconstruction = discriminator(data_var)
            discriminator_realloss = criterion(real_reconstruction, data_var)
            # discriminator_realloss = torch.mean(torch.abs(real_reconstruction - data_var))

            aux.noise.data.resize_(batch_size, aux.nz, 1, 1)
            aux.noise.data.uniform_(-1, 1)
            noise_d_sample = generator(aux.noise).detach()
            noise_reconstruction = discriminator(noise_d_sample)
            discriminator_genloss = criterion(noise_reconstruction, noise_d_sample)
            # discriminator_genloss = torch.mean(torch.abs(noise_reconstruction - noise_d_sample))
            err_discriminator = discriminator_realloss - k_var * discriminator_genloss
            err_discriminator.backward()

            for p in generator.parameters():
                p.requires_grad = True
            optimizerG.zero_grad()

            aux.noise.data.uniform_(-1, 1)
            noise_g_sample = generator(aux.noise)
            discriminator_sample = discriminator(noise_g_sample)
            generator_loss = criterion(noise_g_sample, discriminator_sample.detach())
            # generator_loss = torch.mean(torch.abs(noise_g_sample - discriminator_sample.detach()))
            generator_loss.backward()
                

            #############################
            #     Weight updates
            optimizerG.step()
            optimizerD.step()

            #############################

            equilibrium = (gamma * discriminator_realloss - generator_loss).detach()
            k_var += feedback_control * equilibrium
            k_var.data.clamp_(opt.lb, 1.0)

            if i % 10 == 0:
                global_measure = discriminator_realloss  + torch.abs(equilibrium)
                format_str = ('{} | [{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f}, D(x): {:.4f}, D(G(z)): {:.4f}'
                              ', Global: {:.4f}, k: {:.4f}')
                print(format_str.format(
                    datetime.now().time(),
                    epoch, opt.niter, i, len(dataloader),
                    err_discriminator.data[0],
                    generator_loss.data[0],
                    discriminator_realloss.data[0],
                    discriminator_genloss.data[0],
                    global_measure.data[0], k_var.data[0]))

            if i % 125 == 0:
                vutils.save_image(data_var.data, '%s/%03d_real_samples.png' % (opt.outf, epoch))
                vutils.save_image(real_reconstruction.data, '%s/%03d_real_reconstruction.png' % (opt.outf, epoch))

                fake = generator(aux.fixed_noise)
                vutils.save_image(fake.data, '%s/%03d_fake_samples.png' % (opt.outf, epoch))

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
                                   # local_transforms.RandomRotation(),
                                   transforms.RandomCrop(opt.imageSize),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                               ]))
# transforms.Normalize(mean=(0.3683006, 0.4098993 ,0.5039812),
#                      std=(0.0123775,  0.01128755, 0.01031650)),
# Mean: R: 0.5039812792256271, G: 0.40989934259960137, B: 0.3683006199917145
# Std: R: 0.01237758766377634, G: 0.011287555487577814, B: 0.010316500128800232

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
        fixed_noise = torch.FloatTensor(batch_size, latent_dim, 1, 1).uniform_(-1, 1)
        mean_tensor = torch.FloatTensor(3, opt.imageSize, opt.imageSize)
        label = torch.FloatTensor(batch_size)

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
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    ngpu = int(opt.ngpu)

    discriminator, generator = resnet18(ngpu, ngf, ndf, nz)
    if opt.gen != '':
        generator.load_state_dict(torch.load(opt.gen))
    print(generator)
    if opt.dis != '':
        discriminator.load_state_dict(torch.load(opt.dis))
    print(discriminator)
    """
    discriminator = dcgan.DiscriminatorUp(ngpu, ngf, ndf, nc, nz)
    discriminator.apply(model.weights_init)
    print(discriminator)
    generator = dcgan.GeneratorUp(ngpu, ngf, nc, nz)
    generator.apply(model.weights_init)
    print(generator)
    """

    if opt.cuda:
        discriminator.cuda()
        generator.cuda()
        discriminator = nn.DataParallel(discriminator)
        generator = nn.DataParallel(generator)

    return generator, discriminator


if __name__ == '__main__':
    main()
    # test()
