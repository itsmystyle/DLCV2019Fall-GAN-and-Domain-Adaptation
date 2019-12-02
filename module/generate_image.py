import os
import argparse

import torch
import torchvision.utils as vutils

from utils import set_random_seed
from module.dcgan.generator import Generator as DCGen
from module.acgan.generator import Generator as ACGen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GAN and ACGAN images.")
    parser.add_argument("--output_dir", type=str, help="Directory to store generated images.")
    parser.add_argument("--dcgan_model", type=str, help="Path to DCGan model.")
    parser.add_argument("--acgan_model", type=str, help="Path to ACGan model.")
    parser.add_argument("--random_seed", nargs="+", default=None, help="random seed.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fig_1_path = os.path.join(args.output_dir, "fig1_2.jpg")
    fig_2_path = os.path.join(args.output_dir, "fig2_2.jpg")

    dcgen = DCGen(latent_dim=128)
    dcgen.load_state_dict(torch.load(args.dcgan_model))
    dcgen.to(device)
    acgen = ACGen(latent_dim=120, n_feature_maps=128, embedding_dim=4, n_attributes=1)
    acgen.load_state_dict(torch.load(args.acgan_model))
    acgen.to(device)

    # prepare dc noise
    set_random_seed(int(args.random_seed[0]))
    dc_noise = torch.randn(64, 128, 1, 1, device=device)
    dc_noise[1], dc_noise[2], dc_noise[5] = dc_noise[42], dc_noise[50], dc_noise[54]
    dc_noise[9], dc_noise[18], dc_noise[28] = dc_noise[60], dc_noise[32], dc_noise[34]
    dc_noise[27] = dc_noise[63]
    dc_noise = dc_noise[:32]

    # prepare ac noise
    set_random_seed(int(args.random_seed[1]))
    tmp_noise = torch.randn(28, 120, 1, 1, device=device)
    ac_noise = torch.cat((tmp_noise[4:-8], tmp_noise[-4:]))
    ac_attribute = torch.randint(low=0, high=2, size=(20, 1), dtype=torch.long, device=device,)
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

    vutils.save_image(fake, fig_1_path, nrow=8, padding=2, normalize=True)

    # Generate AC Gan images
    acgen.eval()

    with torch.no_grad():
        fake = acgen(ac_noise, ac_attribute).detach().cpu()

    vutils.save_image(fake, fig_2_path, nrow=4, padding=2, normalize=True, scale_each=True)
