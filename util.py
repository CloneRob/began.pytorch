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
