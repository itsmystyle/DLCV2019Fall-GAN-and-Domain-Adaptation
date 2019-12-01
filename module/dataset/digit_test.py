import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from module.dataset.utils import get_transform


class DigitTestDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.data = glob.glob(os.path.join(self.images_dir, "*"))
        self.data = sorted(self.data)

        if transform:
            self.transform = transform
        else:
            self.transform = get_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]
        image = Image.open(image_path)
        image = self.transform(image)

        return image, image_path.split("/")[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Digit dataset.")

    parser.add_argument("images_dir", type=str, help="Path to images stored directory.")

    args = parser.parse_args()

    dataset = DigitTestDataset(args.images_dir)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    images, images_name = next(iter(dataloader))

    print(images.shape, len(images_name))

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(images[:64], padding=2, normalize=True).cpu(), (1, 2, 0),
        ),
    )
    plt.savefig(os.path.join("./", "example.png"))
    print(images_name)
