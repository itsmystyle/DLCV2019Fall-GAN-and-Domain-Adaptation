import torch
import torch.nn as nn
import torch.nn.utils as U


class Discriminator(nn.Module):
    def __init__(self, n_feature_maps=64, n_class=1):
        super(Discriminator, self).__init__()

        self.n_feature_maps = n_feature_maps
        self.n_class = n_class
        self.conv_blocks = nn.Sequential(
            U.spectral_norm(nn.Conv2d(3, self.n_feature_maps, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            U.spectral_norm(
                nn.Conv2d(
                    self.n_feature_maps, self.n_feature_maps * 2, 4, 2, 1, bias=False
                )
            ),
            nn.BatchNorm2d(self.n_feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            U.spectral_norm(
                nn.Conv2d(
                    self.n_feature_maps * 2,
                    self.n_feature_maps * 4,
                    4,
                    2,
                    1,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(self.n_feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            U.spectral_norm(
                nn.Conv2d(
                    self.n_feature_maps * 4,
                    self.n_feature_maps * 8,
                    4,
                    2,
                    1,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(self.n_feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        # output layer
        self.adv_layer = nn.Sequential(
            U.spectral_norm(nn.Conv2d(self.n_feature_maps * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )
        self.bangs_aux_layer = nn.Sequential(
            U.spectral_norm(nn.Conv2d(self.n_feature_maps * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )
        self.biglips_aux_layer = nn.Sequential(
            U.spectral_norm(nn.Conv2d(self.n_feature_maps * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )
        self.hmakeups_aux_layer = nn.Sequential(
            U.spectral_norm(nn.Conv2d(self.n_feature_maps * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )
        self.hcheekbones_aux_layer = nn.Sequential(
            U.spectral_norm(nn.Conv2d(self.n_feature_maps * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )
        self.male_aux_layer = nn.Sequential(
            U.spectral_norm(nn.Conv2d(self.n_feature_maps * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )
        self.wlipstick_aux_layer = nn.Sequential(
            U.spectral_norm(nn.Conv2d(self.n_feature_maps * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )
        self.smile_aux_layer = nn.Sequential(
            U.spectral_norm(nn.Conv2d(self.n_feature_maps * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )

    def forward(self, X):
        X = self.conv_blocks(X)
        auxs = torch.cat(
            (
                self.bangs_aux_layer(X),
                self.biglips_aux_layer(X),
                self.hmakeups_aux_layer(X),
                self.hcheekbones_aux_layer(X),
                self.male_aux_layer(X),
                self.wlipstick_aux_layer(X),
            ),
            dim=1,
        ).squeeze()
        return (
            self.adv_layer(X),
            auxs,
            self.smile_aux_layer(X),
        )
