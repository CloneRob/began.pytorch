from models.resnet import resnet18

import torch
from torch.autograd import Variable
import torchvision.utils as vutils

def test(ngpu, ngf, ndf, nz, outf):
    """
    simple tets function which tests output size
    """

    resnetdis, resnetgen = resnet18(ngpu, ngf, ndf, nz)
    resnetdis.cuda()
    resnetgen.cuda()
    print(resnetdis)
    dummy_data = torch.cuda.FloatTensor(100, 3, 64, 64)
    dummy_code = torch.cuda.FloatTensor(100, nz, 1, 1)
    print('Input size: ', dummy_data.size())
    print('h size: ', dummy_code.size())

    dummy_var = Variable(dummy_data)
    dummy_codevar = Variable(dummy_code)
    output = resnetdis(dummy_var)
    fake = resnetgen(dummy_codevar)
    print('Output size: ', output.size())
    print('fake size: ', fake.size())
    assert output.size() == dummy_data.size()
    vutils.save_image(fake.data, '%s/%03d_fake_generator.png' % (outf, 1))

def sample(generator, discriminator, auxillary_variables, batch_size = 104, outf='./samples/'):

    auxillary_variables.noise.data.resize_(batch_size, auxillary_variables.nz, 1, 1)
    for i in range(0, 20):
        auxillary_variables.noise.data.normal_(0, 0.7)
        fakeg = generator(auxillary_variables.noise)
        vutils.save_image(fakeg.data, '%s/%03d_fake_generator.png' % (outf, i))
        faked = discriminator(auxillary_variables.noise)
        vutils.save_image(faked.data, '%s/%03d_fake_discriminator.png' % (outf, i))

def normalization_values(data_path, img_size):
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    dataset = dset.ImageFolder(root=data_path,
                               transform=transforms.Compose([
                                   # transforms.Scale(img_size),
                                   transforms.ToTensor(),
                               ]))
    n = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512,
                                             shuffle=False, num_workers=16)
    accum = torch.FloatTensor(3, img_size, img_size)
    accum.zero_()

    for i, (data, _) in enumerate(dataloader, 0):
        accum += data.sum(dim=0)
        if i % 250 == 0:
            print('[{}/{}]'.format(i * 512, n))
    accum /= n
    print('Mean: R: {}, G: {}, B: {}'.format(accum[0].mean(), accum[1].mean(), accum[2].mean()))
    print('Std: R: {}, G: {}, B: {}'.format(accum[0].std(), accum[1].std(), accum[2].std()))
    vutils.save_image(accum, './mean_image.png')

def adjust_lr(optimizer, epoch, initial_lr=0.00005):
    lr = initial_lr * (0.5 ** (epoch // 25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
