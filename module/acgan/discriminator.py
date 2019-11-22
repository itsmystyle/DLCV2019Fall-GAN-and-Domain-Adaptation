import torch.nn as nn
import torch.nn.utils as U


class Discriminator(nn.Module):
    def __init__(self, n_feature_maps=64, n_class=1, n_attributes=7, p=0.1):
        super(Discriminator, self).__init__()

        self.n_feature_maps = n_feature_maps
        self.n_attributes = n_attributes
        self.n_class = n_class
        self.p = p
        self.conv_blocks = nn.Sequential(
            U.spectral_norm(nn.Conv2d(3, self.n_feature_maps, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(self.n_feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            U.spectral_norm(
                nn.Conv2d(
                    self.n_feature_maps, self.n_feature_maps * 2, 4, 2, 1, bias=False
                )
            ),
            nn.BatchNorm2d(self.n_feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(self.p),
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
            nn.Dropout2d(self.p),
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
            nn.Dropout2d(self.p),
            U.spectral_norm(
                nn.Conv2d(
                    self.n_feature_maps * 8, self.n_attributes + 1, 4, 1, 0, bias=False
                )
            ),
            nn.Sigmoid(),
        )

    def forward(self, X):
        X = self.conv_blocks(X)
        X = X.squeeze()

        # return adv_loss, smile_loss
        return X[:, 0], X[:, 1:]
