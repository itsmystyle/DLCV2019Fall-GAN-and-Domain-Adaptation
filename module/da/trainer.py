import os
import argparse

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from module.da.dann import DANN
from module.utils import weights_init, set_random_seed
from module.dataset.digit import DigitDataset
from module.metrics import MulticlassAccuracy, BCAccuracy


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
        lr=1e-3,
        workers=8,
        batch_size=128,
    ):

        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_source_only = train_source_only

        # Models
        self.model = DANN(p=0.5)
        self.model.to(self.device)
        self.model.apply(weights_init)
        print(self.model)

        # Optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=lr)

        # Criterion
        self.criterion = nn.NLLLoss()
        self.domain_criterion = nn.BCELoss()

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
        self.mcAcc = MulticlassAccuracy()
        self.targetAcc = BCAccuracy()
        self.sourceAcc = BCAccuracy()

    def fit(self):
        # train model
        print("===> start training ...")
        iters = 0
        val_iters = 0
        best_acc = 0
        low_valid_loss = 100000.0

        for epoch in range(1, self.epochs + 1):
            iters = self._run_one_epoch(epoch, iters)

            # generate figures
            val_iters, best_acc, low_valid_loss = self._eval_one_epoch(
                epoch, val_iters, best_acc, low_valid_loss
            )

            # save model
            # self.save(os.path.join(self.save_dir, "model_{}.pth.tar".format(epoch)))

        print("Best model Acc:{}, Loss:{}".format(best_acc, low_valid_loss))

    def _run_one_epoch(self, epoch, iters):
        # train model one epoch
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
        self.mcAcc.reset()
        self.targetAcc.reset()
        self.sourceAcc.reset()

        for idx, (sources, targets) in trange:

            # prepare training data
            source_images, labels = sources
            target_images, _ = targets
            bs_source = source_images.shape[0]
            bs_target = target_images.shape[0]
            source_images = source_images.to(self.device)
            labels = labels.to(self.device)
            target_images = target_images.to(self.device)
            source_labels = torch.full((bs_source,), self.source_label, device=self.device)
            target_labels = torch.full((bs_target,), self.target_label, device=self.device)

            p = (idx + start_steps) / total_steps
            Lambda = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            self.optim.zero_grad()

            # calculate label loss
            label_pred, domain_pred = self.model(source_images, Lambda)
            label_loss = self.criterion(label_pred, labels)
            loss = label_loss

            self.mcAcc.update(label_pred, labels)
            LabelLoss.append(label_loss.item())

            if not self.train_source_only:
                # calculate source domain loss
                source_domain_loss = self.domain_criterion(domain_pred.view(-1), source_labels)

                self.sourceAcc.update(
                    domain_pred.view(-1).detach().cpu().numpy(),
                    source_labels.view(-1).cpu().numpy(),
                )

                # calculate target domain loss
                _, domain_pred = self.model(target_images, Lambda)
                target_domain_loss = self.domain_criterion(domain_pred.view(-1), target_labels)

                self.targetAcc.update(
                    domain_pred.view(-1).detach().cpu().numpy(),
                    target_labels.view(-1).cpu().numpy(),
                )

                domain_loss = source_domain_loss + target_domain_loss
                loss += domain_loss

                DomainLoss.append(domain_loss.item())

            loss.backward()
            self.optim.step()

            # update tqdm
            postfix_dict = {
                "L/D": "{:.5f}/{:.5f}".format(
                    np.mean(LabelLoss), np.mean(DomainLoss) if DomainLoss != [] else 0.0
                ),
                "MCAcc.": "{:.5f}".format(self.mcAcc.get_score()),
                "S/T": "{:.5f}/{:.5f}".format(
                    self.sourceAcc.get_score(), self.targetAcc.get_score()
                ),
                "Lmbd": "{:.5f}".format(Lambda),
            }
            trange.set_postfix(**postfix_dict)

            # writer write loss per iteration
            self.writer.add_scalars(
                "Label", {"loss": np.array(LabelLoss).mean()}, iters,
            )
            self.writer.add_scalars(
                "Domain", {"loss": np.array(DomainLoss).mean() if DomainLoss != [] else 0.0}, iters,
            )
            self.writer.add_scalars(
                "Accuracy", {"label": self.mcAcc.get_score()}, iters,
            )

            iters += 1

        # writer write loss per epoch
        self.writer.add_scalars(
            "epoch_loss",
            {
                "label": np.array(LabelLoss).mean(),
                "domain": np.array(DomainLoss).mean() if DomainLoss != [] else 0.0,
            },
            epoch,
        )

        return iters

    def _eval_one_epoch(self, epoch, val_iters, best_acc, low_valid_loss):
        # evaluate model one epoch
        self.model.eval()

        LabelLoss = []
        self.mcAcc.reset()

        trange = tqdm(
            enumerate(self.valid_dataloader), total=len(self.valid_dataloader), desc="Valid",
        )

        with torch.no_grad():
            for idx, (images, labels) in trange:
                images, labels = images.to(self.device), labels.to(self.device)

                preds, _ = self.model(images, 1.0)
                loss = self.criterion(preds, labels)

                LabelLoss.append(loss.item())
                self.mcAcc.update(preds, labels)

                # update tqdm
                postfix_dict = {
                    "Loss": "{:.5f}".format(np.mean(LabelLoss)),
                    "Acc.": "{:.5f}".format(self.mcAcc.get_score()),
                }
                trange.set_postfix(**postfix_dict)

                # writer write loss per validation iteration
                self.writer.add_scalars(
                    "Val_Label", {"loss": np.array(LabelLoss).mean()}, val_iters,
                )
                self.writer.add_scalars(
                    "Val_Accuracy", {"label": self.mcAcc.get_score()}, val_iters,
                )

                val_iters += 1

            # writer write loss per epoch
            self.writer.add_scalars(
                "epoch_loss", {"label_val": np.array(LabelLoss).mean()}, epoch,
            )

            # save best acc model
            if self.mcAcc.get_score() > best_acc:
                print("Best model saved!")
                self.save(os.path.join(self.save_dir, "model_best.pth.tar"))
                best_acc = self.mcAcc.get_score()

            # save lowest valid loss model
            if np.array(LabelLoss).mean() < low_valid_loss:
                print("Lowest valid loss model saved!")
                self.save(os.path.join(self.save_dir, "model_lowest.pth.tar"))
                low_valid_loss = np.array(LabelLoss).mean()

        return val_iters, best_acc, low_valid_loss

    def save(self, path):
        torch.save(
            self.model.state_dict(), path,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Domain Adaptation trainer.")

    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("model_dir", type=str, help="Directory path to store models.")
    parser.add_argument("source_datapath", type=str, help="Path to source dataset.")
    parser.add_argument("source_images_dir", type=str, help="Path to source images directory.")
    parser.add_argument("valid_datapath", type=str, help="Path to validation dataset.")
    parser.add_argument("valid_images_dir", type=str, help="Path to validation images directory.")
    parser.add_argument("--target_datapath", type=str, help="Path to target dataset.")
    parser.add_argument("--target_images_dir", type=str, help="Path to target images directory.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed.")

    args = parser.parse_args()

    set_random_seed(args.random_seed)
    source_dataset = DigitDataset(args.source_datapath, args.source_images_dir)
    validset = DigitDataset(args.valid_datapath, args.valid_images_dir)
    target_dataset = None
    if args.target_datapath is not None and args.target_images_dir is not None:
        target_dataset = DigitDataset(args.target_datapath, args.target_images_dir)
        train_source_only = False
    else:
        train_source_only = True
        target_dataset = source_dataset
    writer = SummaryWriter(os.path.join(args.model_dir, "train_logs"))

    trainer = Trainer(
        args.epochs,
        source_dataset,
        target_dataset,
        validset,
        train_source_only,
        writer,
        args.model_dir,
        lr=1e-3,
        workers=8,
        batch_size=args.batch_size,
    )

    trainer.fit()
