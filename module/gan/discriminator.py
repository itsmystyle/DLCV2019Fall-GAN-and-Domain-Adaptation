import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_feature_maps=64):
        super(Discriminator, self).__init__()

        self.n_feature_maps = n_feature_maps
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, self.n_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                self.n_feature_maps, self.n_feature_maps * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(self.n_feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                self.n_feature_maps * 2, self.n_feature_maps * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(self.n_feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                self.n_feature_maps * 4, self.n_feature_maps * 8, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(self.n_feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.n_feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, X):
        return self.conv_blocks(X)
