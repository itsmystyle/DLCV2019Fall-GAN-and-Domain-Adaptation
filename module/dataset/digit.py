import os
import argparse

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from module.dataset.utils import get_transform


class DigitDataset(Dataset):
    def __init__(
        self, label_paths, images_dirs, transform=None,
    ):
        self.images_dirs = images_dirs
        self.multiple_domain = True if len(images_dirs) > 1 else False
        datas = []
        for idx, label_path in enumerate(label_paths):
            data = pd.read_csv(label_path)
            data["domain"] = idx
            datas.append(data)
        self.data = pd.concat(datas)

        if transform:
            self.transform = transform
        else:
            self.transform = get_transform()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        domain = self.data.iloc[index].domain
        image_path = os.path.join(
            self.images_dirs[domain], self.data.iloc[index].image_name
        )
        image = Image.open(image_path)
        image = self.transform(image)
        label = self.data.iloc[index].label

        return (image, torch.tensor(label).long(), torch.tensor(domain).long())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MNIST-M dataset.")

    parser.add_argument(
        "--label_paths", nargs="+", help="Paths to load label csv.", required=True
    )
    parser.add_argument(
        "--images_dirs",
        nargs="+",
        help="Paths to images stored directory.",
        required=True,
    )

    args = parser.parse_args()

    dataset = DigitDataset(args.label_paths, args.images_dirs)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    images, labels, domains = next(iter(dataloader))

    print(images.shape, labels.shape, domains.shape)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(images[:64], padding=2, normalize=True).cpu(), (1, 2, 0),
        ),
    )
    plt.savefig(os.path.join("./", "example.png"))
    print(labels)
