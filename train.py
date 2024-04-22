import os

import numpy as np
import torch
from torch import FloatTensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from tqdm import tqdm
import torch.nn.functional as F

from dataset_prep.xray import Xray
from model.discriminator import Discriminator
from model.VITGAN import VITGAN
from torch.utils.tensorboard import SummaryWriter


def train(modelConfig):
    adversarial_loss = torch.nn.MSELoss()
    tb_writer = SummaryWriter(log_dir='runs/xray_3')

    device = torch.device(modelConfig["device"])
    train_transform = transforms.Compose([
        transforms.Resize(modelConfig['img_size']),
        transforms.CenterCrop(modelConfig['img_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = Xray(root=modelConfig['dataset_root'], target_transforms=train_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=modelConfig['batch_size'],
        shuffle=True,
    )

    generator = VITGAN(modelConfig).to(device)
    # optimizerG = torch.optim.Adam(generator.parameters(), lr=modelConfig['lr_g'], weight_decay=1e-4)
    optimizerG = torch.optim.SGD(generator.parameters(), lr=modelConfig['lr_g'])
    generator_test = VITGAN(modelConfig).to(device)

    init_report = torch.zeros((1, 64, 768), device=device)
    init_img = torch.zeros((1, 3, 256, 256), device=device)
    tb_writer.add_graph(generator, [init_img, init_report])

    discriminator = Discriminator().to(device)
    # optimizerD = torch.optim.Adam(discriminator.parameters(), lr=modelConfig['lr_d'], weight_decay=1e-4)
    optimizerD = torch.optim.SGD(discriminator.parameters(), lr=modelConfig['lr_d'])

    eval_noise = torch.randn((1, 3, 256, 256)).cuda()

    for i in range(modelConfig['epoch']):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataloader:
            for (images, report) in tqdmDataloader:
                real_data = images.to(device)
                report = report.to(device).reshape(report.shape[0], 64, 768)

                noise = torch.randn((report.shape[0], 3, 256, 256)).cuda()

                # Adversarial ground truths
                valid = Variable(FloatTensor(real_data.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
                fake = Variable(FloatTensor(real_data.shape[0], 1).fill_(0.0), requires_grad=False).cuda()

                # train generator
                optimizerG.zero_grad()
                generated_img = generator(noise, report)

                validity = discriminator(report, generated_img)
                errG = adversarial_loss(validity, valid)

                errG.backward()
                optimizerG.step()

                # train discriminator
                optimizerD.zero_grad()

                validity_real = discriminator(report, real_data)
                errD_real = adversarial_loss(validity_real, valid)
                errD_real.backward(retain_graph=True)

                validity_fake = discriminator(report, generated_img.detach())
                errD_fake = adversarial_loss(validity_fake, fake)
                errD_fake.backward()

                errD = (errD_fake + errD_real) / 2
                # errD.backward()
                optimizerD.step()

                tqdmDataloader.set_postfix(ordered_dict={
                    "epoch": i,
                    "loss_G": errG.data,
                    "loss_D_real": errD_real.data,
                    "loss_D_fake": errD_fake.data,
                    "loss_D": errD
                })

            tb_writer.add_scalar("errG", scalar_value=errG.data, global_step=i)
            tb_writer.add_scalar("errD", errD.data, i)
            tb_writer.add_scalar("errD_fake", errD_fake.data, i)
            tb_writer.add_scalar("errD_real", errD_real.data, i)

            torch.save(discriminator.state_dict(), os.path.join(
                modelConfig['save_weight_dir'], 'discriminator/ckpt_' + str(i) + "_.pt"
            ))
            torch.save(generator.state_dict(), os.path.join(
                modelConfig['save_weight_dir'], 'generator/ckpt_' + str(i) + "_.pt"
            ))

        with torch.no_grad():
            generator_test.eval()
            torch.cuda.manual_seed(100)
            ckpt_G = torch.load(os.path.join(
                modelConfig['save_weight_dir'], 'generator/ckpt_' + str(i) + "_.pt"), map_location=device)
            generator_test.load_state_dict(ckpt_G)
            report = torch.tensor(np.load("test_label.npy")).cuda().reshape(1, 64, 768)

            generated_img = generator(eval_noise, report).reshape(3, 256, 256).to(device)

            save_image(generated_img, os.path.join(modelConfig['save_img'], f'{str(i)}.jpg'))
            tb_writer.add_image("generate img", generated_img, i, dataformats="CHW")

    tb_writer.close()
