import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from module.utils import set_random_seed
from module.dcgan.generator import Generator as DCGen
from module.acgan.generator import Generator as ACGen

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GAN and ACGAN images.")
    parser.add_argument(
        "--output_dir", type=str, help="Directory to store generated images."
    )
    parser.add_argument("--dcgan_model", type=str, help="Path to DCGan model.")
    parser.add_argument("--acgan_model", type=str, help="Path to ACGan model.")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size.")

    args = parser.parse_args()

    set_random_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fig_1_path = os.path.join(args.output_dir, "fig1_2.jpg")
    fig_2_path = os.path.join(args.output_dir, "fig2_2.jpg")

    dcgen = DCGen(latent_dim=128)
    dcgen.to(device)
    acgen = ACGen(latent_dim=120, n_feature_maps=128, embedding_dim=4, n_attributes=1)
    acgen.to(device)

    dc_noise = torch.randn(32, 128, 1, 1, device=device)
    ac_noise = torch.randn(20, 120, 1, 1, device=device)
    ac_attribute = torch.randint(
        low=0, high=2, size=(20, 1), dtype=torch.long, device=device,
    )
    ac_attribute[0, -1] = 0
    for i in range(1, 20):
        if i % 2 != 0:
            ac_attribute[i] = ac_attribute[i - 1]
            ac_attribute[i, -1] = 1
            ac_noise[i] = ac_noise[i - 1]
        else:
            ac_attribute[i, -1] = 0

    # Generate DC Gan images
    dcgen.eval()

    with torch.no_grad():
        fake = dcgen(dc_noise).detach().cpu()

    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.imshow(
        np.transpose(
            vutils.make_grid(fake, padding=2, normalize=True, scale_each=True).cpu(),
            (1, 2, 0),
        ),
    )
    plt.savefig(fig_1_path)
    plt.close()

    # Generate AC Gan images
    acgen.eval()

    with torch.no_grad():
        fake = acgen(ac_noise, ac_attribute).detach().cpu()

    plt.figure(figsize=(5, 2))
    plt.axis("off")
    plt.imshow(
        np.transpose(
            vutils.make_grid(fake, padding=2, normalize=True, scale_each=True).cpu(),
            (1, 2, 0),
        ),
    )
    plt.savefig(fig_2_path)
    plt.close()
