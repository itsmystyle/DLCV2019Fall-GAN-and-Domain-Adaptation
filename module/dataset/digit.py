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
        self, label_path, images_dir, transform=None,
    ):
        self.images_dir = images_dir
        self.data = pd.read_csv(label_path)

        if transform:
            self.transform = transform
        else:
            self.transform = get_transform()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.data.iloc[index].image_name)
        image = Image.open(image_path)
        image = self.transform(image)
        label = self.data.iloc[index].label

        return (image, torch.tensor(label).long())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Digit dataset.")

    parser.add_argument("label_path", type=str, help="Path to load label csv.")
    parser.add_argument("images_dir", type=str, help="Path to images stored directory.")

    args = parser.parse_args()

    dataset = DigitDataset(args.label_path, args.images_dir)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    images, labels = next(iter(dataloader))

    print(images.shape, labels.shape)

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
