import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_feature_maps=64):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.n_feature_maps = n_feature_maps

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.n_feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.n_feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                self.n_feature_maps * 8, self.n_feature_maps * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(self.n_feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                self.n_feature_maps * 4, self.n_feature_maps * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(self.n_feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.n_feature_maps * 2, self.n_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.n_feature_maps, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, X):
        return self.conv_blocks(X)
