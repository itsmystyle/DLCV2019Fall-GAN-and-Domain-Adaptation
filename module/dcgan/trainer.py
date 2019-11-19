import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader

from module.dcgan.generator import Generator
from module.dcgan.discriminator import Discriminator
from module.utils import weights_init


class Trainer:
    def __init__(
        self,
        epochs,
        dataset,
        writer,
        save_dir,
        lr=2e-4,
        beta1=0.5,
        workers=4,
        batch_size=128,
    ):

        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = 100

        # Models
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        print(self.generator)
        print(self.discriminator)

        # Optimizer
        self.OptimG = optim.Adam(
            self.generator.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        self.OptimD = optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999)
        )

        # Criterion
        self.criterion = nn.BCELoss()

        # Dataloader
        self.batch_size = batch_size
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=workers
        )

        # Utils
        self.writer = writer
        self.save_dir = save_dir
        self.fixed_noise = torch.randn(64, self.latent_dim, 1, 1, device=self.device)
        self.real_label = 1
        self.fake_label = 0

    def fit(self):
        # train model
        print("===> start training ...")
        iters = 0

        for epoch in range(1, self.epochs + 1):
            iters = self._run_one_epoch(epoch, iters)

            # generate figures
            self._eval_one_epoch(epoch)

            # save model
            self.save(self.save_dir, epoch)

    def _run_one_epoch(self, epoch, iters):
        # train generator and discriminator one epoch
        self.generator.train()
        self.discriminator.train()

        trange = tqdm(
            enumerate(self.dataloader),
            total=len(self.dataloader),
            desc="Epoch {}".format(epoch),
        )

        for idx, (images, labels) in trange:

            ###############################
            # Train D with real image
            self.discriminator.zero_grad()
            real_image = images.to(self.device)
            label = torch.full((self.batch_size,), self.real_label, device=self.device)
            output = self.discriminator(real_image)
            errD_real = self.criterion(output.view(-1), label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train D with fake image
            noise = torch.randn(
                self.batch_size, self.latent_dim, 1, 1, device=self.device
            )
            fake = self.generator(noise)
            label.fill_(self.fake_label)
            output = self.discriminator(fake.detach())
            errD_fake = self.criterion(output.view(-1), label)
            errD_fake.backward()

            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake

            self.OptimD.step()
            ###############################

            ###############################
            self.generator.zero_grad()
            label.fill_(self.real_label)
            output = self.discriminator(fake)
            errG = self.criterion(output.view(-1), label)
            errG.backward()

            D_G_z2 = output.mean().item()

            self.OptimG.step()
            ###############################

            # update tqdm
            postfix_dict = {
                "dRerr": "{:.3f}".format(errD_real.item()),
                "dFerr": "{:.3f}".format(errD_fake.item()),
                "dErr": "{:.3f}".format(errD.item()),
                "gErr": "{:.3f}".format(errG.item()),
                "dRm": "{:.3f}".format(D_x),
                "dFm": "{:.3f}".format(D_G_z1),
                "gFm": "{:.3f}".format(D_G_z2),
            }
            trange.set_postfix(**postfix_dict)

            # writer write loss of g and d
            self.writer.add_scalar("loss/d_real_loss", errD_real.item(), iters)
            self.writer.add_scalar("loss/d_fake_loss", errD_fake.item(), iters)
            self.writer.add_scalar("loss/d_loss", errD.item(), iters)
            self.writer.add_scalar("loss/g_loss", errG.item(), iters)
            self.writer.add_scalar("loss/d_fake_mean", D_x, iters)
            self.writer.add_scalar("loss/d_real_mean", D_G_z1, iters)
            self.writer.add_scalar("loss/g_mean", D_G_z2, iters)

            iters += 1

        return iters

    def _eval_one_epoch(self, epoch):
        # plot fixed noise figures
        self.generator.eval()

        with torch.no_grad():
            fake = self.generator(self.fixed_noise).detach().cpu()

        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(fake, padding=2, normalize=True).cpu(), (1, 2, 0),
            ),
        )
        plt.savefig(os.path.join(self.save_dir, "epoch_{}.png".format(epoch)))

    def save(self, epoch):
        torch.save(
            self.generator.state_dict(),
            os.path.join(self.save_dir, "generator_{}.pth.tar".format(epoch)),
        )
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(self.save_dir, "discriminator_{}.pth.tar".format(epoch)),
        )
