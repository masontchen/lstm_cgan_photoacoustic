import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as tr

import os

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_original


class LSTMModel(BaseModel):
    def name(self):
        return 'LSTMModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(dataset_mode='triplet')
        parser.set_defaults(which_model_netG='resnet_6blocks')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_GAN', type=float, default=1, help='weight for GAN loss')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        # load/define networks

        self.netG = networks_original.define_G(64, opt.output_nc, opt.ngf,
                                       'lstm', opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, hidden_channels=[64], generator=opt.generator)

        if self.isTrain:
            use_sigmoid = False
            self.netD = networks_original.define_D(opt.input_nc * 3 + opt.output_nc, opt.ndf,
                                                   opt.which_model_netD,
                                                   opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks_original.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A1 = input['A1' if AtoB else 'B'].to(self.device)
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_A3 = input['A3' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A1, self.real_A, self.real_A3)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A1, self.real_A, self.real_A3, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A1, self.real_A, self.real_A3, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A1, self.real_A, self.real_A3, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        # self.loss_msssim = 1 - pytorch_msssim.msssim(self.fake_B, self.real_B, normalize=True)
        # self.loss_G = 0.84 * self.loss_msssim + 0.16 * self.loss_G_L1

        self.loss_G = self.loss_G_GAN * self.opt.lambda_GAN + self.loss_G_L1 * self.opt.lambda_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        # self.optimizer_G1.zero_grad()
        # self.optimizer_Glstm.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # self.optimizer_G1.step()
        # self.optimizer_Glstm.step()
