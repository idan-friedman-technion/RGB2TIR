# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Import utils directory
import sys
import os
import shutil
from os import path
import pathlib

################################################################

import argparse
import itertools
import statistics
from PIL import Image

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np

import torch
import tqdm
# import lpips
# import matplotlib.pylab as plot


from RGB2TIR.utils.models import Generator
from RGB2TIR.utils.models import Discriminator
from RGB2TIR.utils.utils import ReplayBuffer
from RGB2TIR.utils.utils import LambdaLR
from RGB2TIR.utils.vgg_loss import PerceptualLoss
# from RGB2TIR.utils.utils import LoggerS
from RGB2TIR.utils.utils import weights_init_normal
from RGB2TIR.utils.utils import ssim as criterion_ssim
from RGB2TIR.utils.datasets import ImageDataset
from RGB2TIR.utils.datasets import shuffle_data
from RGB2TIR.utils.datasets import get_list_of_files

main_dir = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(main_dir))

# default starting vals for train
train_arguments = {
    "epoch": 0,  # starting epoch
    "n_epochs": 50,
    "batchSize": 1,
    "lr": 0.005,
    "decay_epoch": 20,
    "size": 256,
    "TIR_w": 640,
    "TIR_h": 480,
    "RGB_w": 1280,
    "RGB_h": 960,
    "input_nc": 3,
    "output_nc": 1,
    "cuda": "store_true",
    "nr": "store_true",
    "create_new_net": "store_true",
}


class Full_net_obj():
    def __init__(self, opt, n_epochs=0, batchSize=0):
        self.opt = opt
        self.n_epochs = n_epochs if (n_epochs != 0) else opt.n_epochs
        self.batchSize = batchSize if (batchSize != 0) else opt.batchSize

        if torch.cuda.is_available() and not self.opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        ###### shuffle data ######
        if self.opt.sd:
            shuffle_data()

        ###### Definition of variables ######
        # Networks
        # vgg19_critertion = torchvision.models.vgg19_bn(pretrained=True)
        self.vgg19_criterion = PerceptualLoss(features_to_compute=['conv5_4'], criterion=torch.nn.L1Loss(),
                                              shave_edge=None)
        # self.criterion_lpips = lpips.LPIPS(net='alex', version=0.1)

        self.netG_TIR_2_RGB = Generator(self.opt.input_nc, self.opt.output_nc, input_type="TIR")
        self.netG_RGB_2_TIR = Generator(self.opt.output_nc, self.opt.input_nc, input_type="RGB")
        self.netD_TIR = Discriminator(self.opt.input_nc, input_type="TIR")
        self.netD_RGB = Discriminator(self.opt.output_nc, input_type="RGB")

        if self.opt.cuda:
            self.vgg19_criterion.cuda()
            # self.criterion_lpips.cuda()
            self.netG_TIR_2_RGB.cuda()
            self.netG_RGB_2_TIR.cuda()
            self.netD_TIR.cuda()
            self.netD_RGB.cuda()

        # Lossess
        self.criterion_MSE = torch.nn.MSELoss()
        self.criterion_L1 = torch.nn.L1Loss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if self.opt.cuda else torch.Tensor
        self.input_TIR = Tensor(self.batchSize, self.opt.input_nc, self.opt.TIR_w, self.opt.TIR_h)
        self.input_RGB = Tensor(self.batchSize, self.opt.output_nc, self.opt.RGB_w, self.opt.RGB_h)
        self.target_real = Variable(Tensor(self.batchSize).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(self.batchSize).fill_(0.0), requires_grad=False)

        self.fake_TIR_buffer = ReplayBuffer()
        self.fake_RGB_buffer = ReplayBuffer()

        # Dataset loader
        self.train_TIR_transforms_ = [transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))]
        self.train_RGB_transforms_ = [transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.test_TIR_transforms_ = [transforms.ToTensor()]
        self.test_RGB_transforms_ = [transforms.ToTensor()]


    def initialize_nets(self, create_new_net=True):
        if os.path.exists(os.path.join(main_dir,
                                       self.opt.generator_TIR_2_RGB)) and not create_new_net:  # load pre trained networks
            print("Loading exists nets...")
            self.netG_TIR_2_RGB.load_state_dict(torch.load(os.path.join(main_dir, self.opt.generator_TIR_2_RGB)))
            self.netG_RGB_2_TIR.load_state_dict(torch.load(os.path.join(main_dir, self.opt.generator_RGB_2_TIR)))
            self.netD_TIR.load_state_dict(torch.load(os.path.join(main_dir, self.opt.discriminator_TIR)))
            self.netD_RGB.load_state_dict(torch.load(os.path.join(main_dir, self.opt.discriminator_RGB)))

        else:
            self.netG_TIR_2_RGB.apply(weights_init_normal)
            self.netG_RGB_2_TIR.apply(weights_init_normal)
            self.netD_TIR.apply(weights_init_normal)
            self.netD_RGB.apply(weights_init_normal)

    def initialize_optimizers(self, lr=0):
        # Optimizers & LR schedulers
        if lr == 0:
            lr = self.opt.lr
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_TIR_2_RGB.parameters(), self.netG_RGB_2_TIR.parameters()),
            lr=lr, betas=(0.5, 0.999))
        self.optimizer_D_TIR = torch.optim.Adam(self.netD_TIR.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D_RGB = torch.optim.Adam(self.netD_RGB.parameters(), lr=lr, betas=(0.5, 0.999))

        # self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G,
        #                                                         lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch,
        #                                                                            self.opt.decay_epoch).step)
        # self.lr_scheduler_D_TIR = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_TIR,
        #                                                             lr_lambda=LambdaLR(self.opt.n_epochs,
        #                                                                                self.opt.epoch,
        #                                                                                self.opt.decay_epoch).step)
        # self.lr_scheduler_D_RGB = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_RGB,
        #                                                             lr_lambda=LambdaLR(self.opt.n_epochs,
        #                                                                                self.opt.epoch,
        #                                                                                self.opt.decay_epoch).step)

    def discriminators_train(self, real_TIR, real_RGB):
        ###### Discriminator TIR ######
        self.optimizer_D_TIR.zero_grad()

        # Real loss
        pred_real = self.netD_TIR(real_TIR)
        loss_D_real = self.criterion_MSE(pred_real[0, :], self.target_real)
        # pred_real_l.append(pred_real[0, :].item())

        # Fake loss
        fake_TIR = self.fake_TIR_buffer.push_and_pop(self.fake_TIR)
        pred_fake = self.netD_TIR(self.fake_TIR.detach())
        loss_D_fake = self.criterion_MSE(pred_fake[0, :], self.target_fake)
        # pred_fake_l.append(pred_fake[0, :].item())

        # Total loss
        loss_D_TIR = (loss_D_real + loss_D_fake) * 0.5
        loss_D_TIR.backward()

        self.optimizer_D_TIR.step()
        ###################################

        ###### Discriminator RGB ######
        self.optimizer_D_RGB.zero_grad()

        # Real loss
        pred_real = self.netD_RGB(real_RGB)
        loss_D_real = self.criterion_MSE(pred_real[0, :], self.target_real)

        # Fake loss
        fake_RGB = self.fake_RGB_buffer.push_and_pop(self.fake_RGB)
        pred_fake = self.netD_RGB(self.fake_RGB.detach())
        loss_D_fake = self.criterion_MSE(pred_fake[0, :], self.target_fake)

        # Total loss
        loss_D_RGB = (loss_D_real + loss_D_fake) * 0.5
        loss_D_RGB.backward()

        self.optimizer_D_RGB.step()
        ###################################

    def generators_train(self, real_TIR, real_RGB, pre_train=False, vgg_w=1, gan_w=1, generate_w=1, cycle_w=1):
        ###### Generators TIR_2_RGB and RGB_2_TIR ######
        self.optimizer_G.zero_grad()

        # GAN loss
        pred_fake = self.netD_RGB(self.fake_RGB)
        loss_GAN_TIR_2_RGB = self.criterion_MSE(pred_fake[0, :], self.target_real) * gan_w
        # Gan_loss.append(loss_GAN_TIR_2_RGB.item())

        pred_fake = self.netD_TIR(self.fake_TIR)
        loss_GAN_RGB_2_TIR = self.criterion_MSE(pred_fake[0, :], self.target_real) * gan_w

        # Fake vs. Real Loss
        # loss_TIR_Generate = self.criterion_L1(self.fake_TIR, real_TIR) * generate_w
        # loss_RGB_Generate = self.criterion_L1(self.fake_RGB, real_RGB) * generate_w
        # Generate_loss.append(loss_RGB_Generate.item())

        # Cycle loss
        recovered_TIR = self.netG_RGB_2_TIR(self.fake_RGB)
        loss_cycle_ABA = self.criterion_L1(recovered_TIR, real_TIR) * cycle_w

        recovered_RGB = self.netG_TIR_2_RGB(self.fake_TIR)
        loss_cycle_BAB = self.criterion_L1(recovered_RGB, real_RGB) * cycle_w
        # Recovered_loss.append(loss_cycle_BAB.item())

        # VGG loss
        loss_vgg_TIR = self.vgg19_criterion(recovered_TIR, real_TIR) * vgg_w
        loss_vgg_RGB = self.vgg19_criterion(recovered_RGB, real_RGB) * vgg_w
        # Vgg_loss.append(loss_vgg_RGB.item())

        # Total loss
        total_loss_GAN = loss_GAN_TIR_2_RGB + loss_GAN_RGB_2_TIR
        # total_loss_Generate = loss_TIR_Generate + loss_RGB_Generate
        total_loss_cycle = loss_cycle_ABA + loss_cycle_BAB
        total_loss_vgg = loss_vgg_TIR + loss_vgg_RGB

        loss_G = total_loss_cycle + total_loss_vgg
        if not pre_train:
            loss_G += total_loss_GAN

        loss_G.backward()

        self.optimizer_G.step()

    def train(self, pre_train=False, vgg_w=40, gan_w=1, generate_w=1, cycle_w=1):
        dataloader = DataLoader(
            ImageDataset(TIR_transforms_=self.test_TIR_transforms_, RGB_transforms_=self.test_RGB_transforms_, unaligned=True),
            batch_size=self.batchSize, shuffle=True, num_workers=0)

        n_epochs = 1 if (pre_train) else self.opt.n_epochs
        print('#########################################')
        print('train')
        print('=====')
        if pre_train:
            print('Statring pre train...')
        else:
            print(f"Starting train. {n_epochs} epoches will be ran, {len(dataloader)} files for each epoch\n")

        for ep in range(n_epochs):
            #print(f"\r\repoch number {ep + 1}/{n_epochs} - {(ep + 1) * 100 / n_epochs}%\n")
            # print(f"\nepoch number: {ep+1} out of {n_epochs}\n")
            for i, batch in enumerate(dataloader):
                if pre_train and i == 100:
                    break
                # item_RGB = batch['RGB'][:, :, 113:1393, 33:993]
                # item_RGB = item_RGB[:,280:1000, 210:750]
                item_RGB = batch['RGB']

                for i in [0, 1]:
                    for j in [0, 1]:
                        Q_TIR = batch['TIR'][0, :, i*320:(i+1)*320, j*240:(j+1)*240]
                        Q_RGB = item_RGB[0, :, i*360:(i+1)*360, j*272:(j+1)*272 ]
                        real_TIR = Variable(self.input_TIR.copy_(Q_TIR[:, :, :]))
                        real_RGB = Variable(self.input_RGB.copy_(Q_RGB[:, :, :]))


                        self.fake_RGB = self.netG_TIR_2_RGB(real_TIR)
                        self.fake_TIR = self.netG_RGB_2_TIR(real_RGB)
                        self.generators_train(real_TIR, real_RGB, pre_train=pre_train, vgg_w=vgg_w, gan_w=gan_w, generate_w=generate_w, cycle_w=cycle_w)
                        self.discriminators_train(real_TIR, real_RGB)

            # Save models checkpoints
            torch.save(self.netG_TIR_2_RGB.state_dict(), os.path.join(main_dir, 'RGB2TIR/output/p_netG_TIR_2_RGB.pth'))
            torch.save(self.netG_RGB_2_TIR.state_dict(), os.path.join(main_dir, 'RGB2TIR/output/p_netG_RGB_2_TIR.pth'))
            torch.save(self.netD_TIR.state_dict(), os.path.join(main_dir, 'RGB2TIR/output/p_netD_TIR.pth'))
            torch.save(self.netD_RGB.state_dict(), os.path.join(main_dir, 'RGB2TIR/output/p_netD_RGB.pth'))

        if pre_train:
            print("Finished Pre-Training")
        else:
            print(f"finished main Training")
        return

    def test(self, TIRtoRGB_dir="TIRtoRGB", RGBtoTIR_dir="RGBtoTIR"):
        self.netG_TIR_2_RGB.eval()
        self.netG_RGB_2_TIR.eval()
        self.netD_RGB.eval()
        dataloader = DataLoader(
            ImageDataset(TIR_transforms_=self.test_TIR_transforms_, RGB_transforms_=self.test_TIR_transforms_, mode='test',
                         unaligned=True), batch_size=self.batchSize, shuffle=True, num_workers=0)

        # Create output dirs if they don't exist
        if os.path.exists(os.path.join(main_dir, f"RGB2TIR/output/{RGBtoTIR_dir}")):
            shutil.rmtree(os.path.join(main_dir, f"RGB2TIR/output/{RGBtoTIR_dir}"))
        os.makedirs(os.path.join(main_dir, f"RGB2TIR/output/{RGBtoTIR_dir}"))
        if os.path.exists(os.path.join(main_dir, f"RGB2TIR/output/{TIRtoRGB_dir}")):
            shutil.rmtree(os.path.join(main_dir, f"RGB2TIR/output/{TIRtoRGB_dir}"))
        os.makedirs(os.path.join(main_dir, f"RGB2TIR/output/{TIRtoRGB_dir}"))

        Gan_loss      = []
        Vgg_loss      = []
        Generate_loss = []

        # print('#########################################')
        #print('test')
        #print('====')
        print(f"Starting test. {len(dataloader)} files to test")

        for i, batch in enumerate(dataloader):
            if i % 10 != 0:
                continue
            # Set model input
            item_RGB = batch['RGB']

            for i in [0, 1]:
                for j in [0, 1]:
                    Q_TIR = batch['TIR'][0, :, i * 320:(i + 1) * 320, j * 240:(j + 1) * 240]
                    Q_RGB = item_RGB[0, :, i * 320:(i + 1) * 320, j * 240:(j + 1) * 240]
                    real_TIR = Variable(self.input_TIR.copy_(Q_TIR[:, :, :]))
                    real_RGB = Variable(self.input_RGB.copy_(Q_RGB[:, :, :]))


                    # Generate output
                    # fake_RGB = netG_TIR_2_RGB(real_TIR)
                    fake_RGB = self.netG_TIR_2_RGB(real_TIR)
                    fake_TIR = self.netG_RGB_2_TIR(real_RGB)

                    # Check empirical results
                    pred_fake = self.netD_RGB(fake_RGB)
                    loss_GAN_TIR_2_RGB = self.criterion_MSE(pred_fake[0, :], self.target_real)
                    Gan_loss.append(loss_GAN_TIR_2_RGB.item())

                    loss_RGB_Generate = self.criterion_L1(fake_RGB, real_RGB)
                    Generate_loss.append(loss_RGB_Generate.item())

                    loss_vgg_RGB = self.vgg19_criterion(fake_RGB, real_RGB)
                    Vgg_loss.append(loss_vgg_RGB.item())

                    fake_RGB = 0.5 * (fake_RGB + 1.0)
                    fake_TIR = 0.5 * (fake_TIR + 1.0)

                    # Save image files
                    # save_image(fake_TIR[0], os.path.join(main_dir, f'RGB2TIR/output/RGBtoTIR/%04d.png') % (i+1))
                    # save_image(fake_RGB[0], os.path.join(main_dir, 'RGB2TIR/output/TIRtoRGB/%04d.png') % (i+1))

                    save_image(fake_TIR[0], os.path.join(main_dir, f"RGB2TIR/output/{RGBtoTIR_dir}/{batch['output'][0]}"))
                    save_image(fake_RGB[0], os.path.join(main_dir, f"RGB2TIR/output/{TIRtoRGB_dir}/{batch['output'][0]}"))

        print('Test is finished')
        return statistics.mean(Vgg_loss), statistics.mean(Generate_loss), statistics.mean(Gan_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.005, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=10,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--TIR_w', type=int, default=320, help='size of the data TIR width (squared assumed)')
    parser.add_argument('--TIR_h', type=int, default=240, help='size of the data crop (squared assumed)')
    parser.add_argument('--RGB_w', type=int, default=360, help='size of the data TIR width (squared assumed)')
    parser.add_argument('--RGB_h', type=int, default=272, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--sd', action='store_true', help='shuffle data for test and train')
    parser.add_argument('--nr', action='store_true', help='don\'t run the network')
    parser.add_argument('--create_new_net', action='store_true',
                        help='create new network, and don\'t work with old one')
    parser.add_argument('--generator_TIR_2_RGB', type=str, default='RGB2TIR/output/p_netG_TIR_2_RGB.pth',
                        help='TIR_2_RGB generator checkpoint file')
    parser.add_argument('--generator_RGB_2_TIR', type=str, default='RGB2TIR/output/p_netG_RGB_2_TIR.pth',
                        help='RGB_2_TIR generator checkpoint file')
    parser.add_argument('--discriminator_TIR', type=str, default='RGB2TIR/output/p_netD_TIR.pth',
                        help='TIR discriminator checkpoint file')
    parser.add_argument('--discriminator_RGB', type=str, default='RGB2TIR/output/p_netD_RGB.pth',
                        help='RGB discriminator checkpoint file')
    opt = parser.parse_args()
    print(opt)

    # TIR_2_RGB_net = Full_net_obj(opt)
    # TIR_2_RGB_net.initialize_nets(create_new_net=False)
    # Vgg, Generate, Gan = TIR_2_RGB_net.test(TIRtoRGB_dir=f"final_TIR_2_RGB_2",
    #                                         RGBtoTIR_dir=f"final_RGB_2_TIR_2")

    lr = 0.0017
    vgg_w, generate_w, gan_w, cycle_w = 10, 0, 7, 3

    TIR_2_RGB_net = Full_net_obj(opt)
    TIR_2_RGB_net.initialize_nets(opt.create_new_net)
    TIR_2_RGB_net.initialize_optimizers(lr=lr)
    TIR_2_RGB_net.train(pre_train=True, gan_w=gan_w, generate_w=generate_w, cycle_w=cycle_w, vgg_w=vgg_w)

    Vgg = []
    Gan = []
    Generate = []
    for i in range(12):
        print(f"Starting loop number {i+1}")
        TIR_2_RGB_net.train(pre_train=False, gan_w=gan_w, generate_w=generate_w, cycle_w=cycle_w, vgg_w=vgg_w)

        Vgg_l, Generate_l, Gan_l = TIR_2_RGB_net.test(TIRtoRGB_dir=f"a1_q_{i}_TIR_2_RGB",
                                                    RGBtoTIR_dir=f"a1_q_{i}_RGB_2_TIR")
        Vgg.append(Vgg_l)
        Gan.append(Gan_l)
        Generate.append(Generate_l)

    print(f"\nVGG Loss -\n{Vgg}")
    print(f"\nGenerate Loss -\n{Generate}")
    print(f"\nGan Loss -\n{Gan}")


    # plot.figure(1)
    # plot.subplot(2, 2, 1)
    # plot.plot(range(12), Vgg)
    # plot.title('Mean Vgg Loss')
    # plot.subplot(2, 2, 2)
    # plot.plot(range(12), Generate)
    # plot.title('Mean Generate Loss')
    # plot.subplot(2, 2, 3)
    # plot.plot(range(12), Gan)
    # plot.title('Mean Gan Loss')


    # lr_vgg_vgg_loss = np.zeros((25, 3))
    # lr_vgg_gan_loss = np.zeros((25, 3))
    # lr_vgg_generate_loss = np.zeros((25, 3))
    # for i in range(25):
    #     lr = 1.7
    #     vgg_w = np.round(np.random.uniform(low=2.5, high=7.5), 1)
    #     print(f"starting loop with: lr = {lr} and vgg_w = {vgg_w}")
    #     TIR_2_RGB_net = Full_net_obj(opt)
    #     TIR_2_RGB_net.initialize_nets(opt.create_new_net)
    #     TIR_2_RGB_net.initialize_optimizers(lr=lr)
    #     TIR_2_RGB_net.train(pre_train=True, vgg_w=vgg_w)
    #     TIR_2_RGB_net.train(pre_train=False, vgg_w=vgg_w)
    #     Vgg, Generate, Gan = TIR_2_RGB_net.test(TIRtoRGB_dir=f"TIR_2_RGB_lr_{lr}_vgg_w_{vgg_w}",
    #                                             RGBtoTIR_dir=f"RGB_2_TIR_lr_{lr}_vgg_w_{vgg_w}")
    #     lr_vgg_vgg_loss[i, :] = lr, vgg_w, Vgg
    #     lr_vgg_gan_loss[i, :] = lr, vgg_w, Gan
    #     lr_vgg_generate_loss[i, :] = lr, vgg_w, Generate
    #
    # # sort by vals
    # lr_vgg_vgg_loss = lr_vgg_vgg_loss[lr_vgg_vgg_loss[:, 2].argsort()]
    # lr_vgg_gan_loss = lr_vgg_gan_loss[lr_vgg_gan_loss[:, 2].argsort()]
    # lr_vgg_generate_loss = lr_vgg_generate_loss[lr_vgg_generate_loss[:, 2].argsort()]
    #
    # # put in csv
    # np.savetxt('lr_vs_vggw_Vgg_loss.csv', lr_vgg_vgg_loss, delimiter="\t")
    # np.savetxt('lr_vs_vggw_Generate_loss.csv', lr_vgg_generate_loss, delimiter="\t")
    # np.savetxt('lr_vs_vggw_Gan_loss.csv', lr_vgg_gan_loss, delimiter="\t")


    # ganw_vs_genw_vs_cyclew_vgg_loss = np.zeros((30, 4))
    # ganw_vs_genw_vs_cyclew_gan_loss = np.zeros((30, 4))
    # ganw_vs_genw_vs_cyclew_generate_loss = np.zeros((30, 4))
    # lr = 0.0017
    # for i in range(30):
    #     gan_w = np.round(np.random.uniform(low=1, high=50), 1)
    #     generate_w = np.round(np.random.uniform(low=1, high=50), 1)
    #     cycle_w = np.round(np.random.uniform(low=1, high=50), 1)
    #     print(f"starting loop with: gan_w = {gan_w} and generate_w = {generate_w} and cycle_w = {cycle_w}")
    #     TIR_2_RGB_net = Full_net_obj(opt)
    #     TIR_2_RGB_net.initialize_nets(opt.create_new_net)
    #     TIR_2_RGB_net.initialize_optimizers(lr=lr)
    #     TIR_2_RGB_net.train(pre_train=True, gan_w=gan_w, generate_w=generate_w, cycle_w=cycle_w)
    #     TIR_2_RGB_net.train(pre_train=False, gan_w=gan_w, generate_w=generate_w, cycle_w=cycle_w)
    #     Vgg, Generate, Gan = TIR_2_RGB_net.test(TIRtoRGB_dir=f"TIR_2_RGB_gan_w_{gan_w}_generate_w_{generate_w}_cycle_w_{cycle_w}",
    #                                             RGBtoTIR_dir=f"RGB_2_TIR_gan_w_{gan_w}_generate_w_{generate_w}_cycle_w_{cycle_w}")
    #     ganw_vs_genw_vs_cyclew_vgg_loss[i, :] = gan_w, generate_w, cycle_w, Vgg
    #     ganw_vs_genw_vs_cyclew_gan_loss[i, :] = gan_w, generate_w, cycle_w, Gan
    #     ganw_vs_genw_vs_cyclew_generate_loss[i, :] = gan_w, generate_w, cycle_w, Generate
    #
    # # sort by vals
    # ganw_vs_genw_vs_cyclew_vgg_loss = ganw_vs_genw_vs_cyclew_vgg_loss[ganw_vs_genw_vs_cyclew_vgg_loss[:, 3].argsort()]
    # ganw_vs_genw_vs_cyclew_gan_loss = ganw_vs_genw_vs_cyclew_gan_loss[ganw_vs_genw_vs_cyclew_gan_loss[:, 3].argsort()]
    # ganw_vs_genw_vs_cyclew_generate_loss = ganw_vs_genw_vs_cyclew_generate_loss[ganw_vs_genw_vs_cyclew_generate_loss[:, 3].argsort()]
    #
    # # put in csv
    # np.savetxt('ganw_vs_genw_vs_cyclew_Vgg_loss_2.csv', ganw_vs_genw_vs_cyclew_vgg_loss, delimiter="\t")
    # np.savetxt('ganw_vs_genw_vs_cyclew_Generate_loss_2.csv', ganw_vs_genw_vs_cyclew_generate_loss, delimiter="\t")
    # np.savetxt('ganw_vs_genw_vs_cyclew_Gan_loss.csv_2', ganw_vs_genw_vs_cyclew_gan_loss, delimiter="\t")




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
