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

from module.da.dann import DANN
from module.utils import weights_init, set_random_seed
from module.dataset.digit import DigitDataset


class Trainer:
    def __init__(
        self,
        epochs,
        source_trainset,
        target_trainset,
        validset,
        train_source_only,
        writer,
        save_dir,
        lr=2e-4,
        workers=8,
        batch_size=128,
    ):

        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_source_only = train_source_only

        # Models
        self.model = DANN()
        self.model.to(self.device)
        self.model.apply(weights_init)
        print(self.model)

        # Optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=lr)

        # Criterion
        self.criterion = nn.NLLLoss()

        # Dataloader
        self.batch_size = batch_size
        self.source_dataloader = DataLoader(
            source_trainset, batch_size=batch_size, shuffle=True, num_workers=workers
        )
        self.target_dataloader = DataLoader(
            target_trainset, batch_size=batch_size, shuffle=True, num_workers=workers
        )
        self.valid_dataloader = DataLoader(
            validset, batch_size=batch_size, shuffle=False, num_workers=workers
        )

        # Utils
        self.writer = writer
        self.save_dir = save_dir
        self.source_label = 0
        self.target_label = 1

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
        self.model.train()

        _len = min(len(self.source_dataloader), len(self.target_dataloader))
        trange = tqdm(
            enumerate(zip(self.source_dataloader, self.target_dataloader)),
            total=_len,
            desc="Epoch {}".format(epoch),
        )

        start_steps = epoch * _len
        total_steps = self.epochs * _len

        DomainLoss = []
        LabelLoss = []

        for idx, (sources, targets) in trange:

            # prepare training data
            source_images, labels = sources
            target_images, _ = targets
            bs = source_images.shape[0]
            source_images = source_images.to(self.device)
            labels = labels.to(self.device)
            target_images = target_images.to(self.device)
            source_labels = torch.full((bs,), self.source_label, device=self.device)
            target_labels = torch.full((bs,), self.target_label, device=self.device)

            Lambda = (
                2.0 / (1.0 + np.exp(-10 * float(idx + start_steps) / total_steps)) - 1
            )

            self.optim.zero_grad()

            # calculate label loss
            label_pred, domain_pred = self.model(source_images, Lambda)
            label_loss = self.criterion(label_pred, labels)
            loss = label_loss

            LabelLoss.append(label_loss.item())

            if not self.train_source_only:
                # calculate source domain loss
                source_domain_loss = self.criterion(domain_pred, source_labels)

                # calculate target domain loss
                _, domain_pred = self.model(target_images, Lambda)
                target_domain_loss = self.criterion(domain_pred, target_labels)

                domain_loss = source_domain_loss + target_domain_loss
                loss += domain_loss

                DomainLoss.append(domain_loss.item())

            loss.backward()
            self.optim.step()

            # update tqdm
            postfix_dict = {
                "Label loss": "{:.5f}".format(np.mean(LabelLoss)),
                "Domain loss": "{:.5f}".format(np.mean(DomainLoss)),
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
                "adv_loss",
                {
                    "D_real": adv_errD_real.item(),
                    "D_fake": adv_errD_fake.item(),
                    "G": adv_errG.item(),
                },
                iters,
            )
            self.writer.add_scalars(
                "aux_loss", {"D": aux_errD_real.item(), "G": aux_errG.item()}, iters,
            )
            self.writer.add_scalars(
                "aux_accuracy",
                {
                    "D_real": self.dAcc_real.get_score(),
                    "D_fake": self.dAcc_fake.get_score(),
                    "G": self.gAcc.get_score(),
                },
                iters,
            )
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
            fake = self.generator(self.fixed_noise, self.fixed_attribute).detach().cpu()

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DCGan trainer.")

    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("model_dir", type=str, help="Directory path to store models.")
    parser.add_argument("attribute_path", type=str, help="Path to load attributes csv.")
    parser.add_argument(
        "--attributes", nargs="+", default=None, help="Attributes use as condition."
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("images_dir", type=str, help="Path to images stored directory.")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed.")

    args = parser.parse_args()

    set_random_seed(args.random_seed)
    if args.attributes is not None:
        dataset = CelebADataset(
            args.attribute_path, args.images_dir, attributes=args.attributes
        )
    else:
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
