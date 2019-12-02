import os
import argparse

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from module.utils import set_random_seed
from module.dataset.digit_test import DigitTestDataset
from module.dsn.dsn import DSN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Adaptation predictor.")

    parser.add_argument("images_dir", type=str, help="Directory to stored images.")
    parser.add_argument("target_domain", type=str, help="Target domain.")
    parser.add_argument(
        "output_path", type=str, help="Output path to store prediction."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed.")

    args = parser.parse_args()

    set_random_seed(args.random_seed)
    dataset = DigitTestDataset(args.images_dir)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model dir
    if args.target_domain == "svhn":
        model_dir = os.path.join("models", "dsn_m2s", "model_best.pth.tar")
    else:
        model_dir = os.path.join("models", "dsn_s2m", "model_best.pth.tar")

    # Models
    model = DSN()
    model.load_state_dict(torch.load(model_dir))
    model.to(device)
    model.eval()

    preds_ls = []
    images_name_ls = []

    with torch.no_grad():
        for idx, (images, images_name) in enumerate(dataloader):
            images = images.to(device)

            preds = model(images, 1.0)
            preds = torch.exp(preds[0]).max(dim=1)[1].detach().cpu().numpy()

            preds_ls.append(preds)
            images_name_ls.append(images_name)

    preds_ls = np.concatenate(preds_ls)
    images_name_ls = np.concatenate(images_name_ls)

    df = pd.DataFrame()
    df["image_name"] = images_name_ls
    df["label"] = preds_ls

    df.to_csv(args.output_path, index=False)
