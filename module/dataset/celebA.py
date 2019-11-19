import os
import argparse

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from module.dataset.utils import get_transform


class CelebADataset(Dataset):
    def __init__(self, attribute_path, images_dir, transform=None):
        self.images_dir = images_dir
        self.data = pd.read_csv(attribute_path)
        self.data = self.data[["image_name", "Smiling"]]
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
        label = self.data.iloc[index].Smiling
        label = torch.tensor(label).long()

        return (image, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CelebA dataset.")
    parser.add_argument("attribute_path", type=str, help="Path to load attributes csv.")
    parser.add_argument("images_dir", type=str, help="Path to images stored directory.")

    args = parser.parse_args()

    dataset = CelebADataset(args.attribute_path, args.images_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    images, labels = next(iter(dataloader))

    print(images.shape, labels.shape)
