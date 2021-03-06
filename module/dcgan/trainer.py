import os
import argparse

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from module.metrics import BCAccuracy
from module.dcgan.generator import Generator
from module.dcgan.discriminator import Discriminator
from module.utils import weights_init, set_random_seed
from module.dataset.celebA import CelebADataset


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
        self.latent_dim = 128

        # Models
        self.generator = Generator(latent_dim=self.latent_dim)
        self.discriminator = Discriminator()
        self.generator.to(self.device)
        self.discriminator.to(self.device)
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

        # Adversarial
        self.d_advAcc_fake = BCAccuracy()
        self.d_advAcc_real = BCAccuracy()
        self.g_advAcc = BCAccuracy()

    def fit(self):
        # train model
        print("===> start training ...")
        iters = 0

        for epoch in range(1, self.epochs + 1):
            iters = self._run_one_epoch(epoch, iters)

            # generate figures
            self._eval_one_epoch(epoch)

            # save model
            self.save(epoch)

    def _run_one_epoch(self, epoch, iters):
        # train generator and discriminator one epoch
        self.generator.train()
        self.discriminator.train()

        trange = tqdm(
            enumerate(self.dataloader),
            total=len(self.dataloader),
            desc="Epoch {}".format(epoch),
        )

        Loss_G = []
        Loss_D = []

        self.d_advAcc_fake.reset()
        self.d_advAcc_real.reset()
        self.g_advAcc.reset()

        for idx, (images, _) in trange:
            bs = images.shape[0]

            ###############################
            # Train D with real image
            self.discriminator.zero_grad()
            real_image = images.to(self.device)
            label = torch.full((bs,), self.real_label, device=self.device)
            output = self.discriminator(real_image)
            errD_real = self.criterion(output.view(-1), label)
            errD_real.backward()

            self.d_advAcc_real.update(
                output.view(-1).detach().cpu().numpy(),
                label.view(-1).detach().cpu().numpy(),
            )

            # Train D with fake image
            noise = torch.randn(bs, self.latent_dim, 1, 1, device=self.device)
            fake = self.generator(noise)
            label.fill_(self.fake_label)
            output = self.discriminator(fake.detach())
            errD_fake = self.criterion(output.view(-1), label)
            errD_fake.backward()

            self.d_advAcc_fake.update(
                output.view(-1).detach().cpu().numpy(),
                label.view(-1).detach().cpu().numpy(),
            )

            errD = errD_real + errD_fake

            self.OptimD.step()
            ###############################

            ###############################
            # Train G and label generated images as real-label
            self.generator.zero_grad()
            label.fill_(self.real_label)
            output = self.discriminator(fake)
            errG = self.criterion(output.view(-1), label)
            errG.backward()

            self.g_advAcc.update(
                output.view(-1).detach().cpu().numpy(),
                label.view(-1).detach().cpu().numpy(),
            )

            self.OptimG.step()
            ###############################

            # save loss
            Loss_G.append(errG.item())
            Loss_D.append(errD.item())

            # update tqdm
            postfix_dict = {
                "LGD": "{:.4f}/{:.4f}".format(
                    np.array(Loss_G).mean(), np.array(Loss_D).mean()
                ),
                "cL_GD": "{:.4f}/{:.4f}".format(errG.item(), errD.item()),
                "AdvGRF": "{:.4f}/{:.4f}/{:.4f}".format(
                    self.g_advAcc.get_score(),
                    self.d_advAcc_real.get_score(),
                    self.d_advAcc_fake.get_score(),
                ),
            }
            trange.set_postfix(**postfix_dict)

            # writer write loss of g and d per iteration
            self.writer.add_scalars(
                "avg_loss",
                {"G": np.array(Loss_G).mean(), "D": np.array(Loss_D).mean()},
                iters,
            )
            self.writer.add_scalars("loss", {"G": errG.item(), "D": errD.item()}, iters)
            self.writer.add_scalars(
                "adv_accuracy",
                {
                    "D_real": self.d_advAcc_real.get_score(),
                    "D_fake": self.d_advAcc_fake.get_score(),
                    "G": self.g_advAcc.get_score(),
                },
                iters,
            )

            iters += 1

        # writer write loss of g and d per epoch
        self.writer.add_scalars(
            "epoch_loss",
            {"G": np.array(Loss_G).mean(), "D": np.array(Loss_D).mean()},
            epoch,
        )

        return iters

    def _eval_one_epoch(self, epoch):
        # plot fixed noise figures
        self.generator.eval()

        with torch.no_grad():
            fake = self.generator(self.fixed_noise).detach().cpu()

        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Fixed Noise Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(fake, padding=2, normalize=True).cpu(), (1, 2, 0),
            ),
        )
        dir = os.path.join(self.save_dir, "figures")
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(os.path.join(dir, "epoch_{}.png".format(epoch)))
        plt.close()

    def save(self, epoch):
        torch.save(
            self.generator.state_dict(),
            os.path.join(self.save_dir, "generator_{}.pth.tar".format(epoch)),
        )
        # torch.save(
        #     self.discriminator.state_dict(),
        #     os.path.join(self.save_dir, "discriminator_{}.pth.tar".format(epoch)),
        # )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DCGan trainer.")

    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("model_dir", type=str, help="Directory path to store models.")
    parser.add_argument("attribute_path", type=str, help="Path to load attributes csv.")
    parser.add_argument(
        "--attributes", nargs="+", default=None, help="Attributes use as condition."
    )
    parser.add_argument("images_dir", type=str, help="Path to images stored directory.")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size.")

    args = parser.parse_args()

    set_random_seed(args.random_seed)
    dataset = CelebADataset(args.attribute_path, args.images_dir)
    writer = SummaryWriter(os.path.join(args.model_dir, "train_logs"))

    trainer = Trainer(
        args.epochs,
        dataset,
        writer,
        args.model_dir,
        lr=2e-4,
        beta1=0.5,
        workers=8,
        batch_size=args.batch_size,
    )

    trainer.fit()
