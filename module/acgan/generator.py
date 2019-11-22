import torch
import torch.nn as nn
import torch.nn.utils as U


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim=100,
        n_feature_maps=64,
        n_class=2,
        embedding_dim=8,
        n_attributes=7,
    ):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.n_feature_maps = n_feature_maps
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.n_attributes = n_attributes

        self.embeds = []
        for i in range(self.n_attributes):
            self.embeds.append(nn.Embedding(self.n_class, self.embedding_dim))
        self.embeds = nn.ModuleList(self.embeds)

        self.conv_blocks = nn.Sequential(
            U.spectral_norm(
                nn.ConvTranspose2d(
                    (self.latent_dim + self.n_attributes * self.embedding_dim),
                    self.n_feature_maps * 8,
                    4,
                    1,
                    0,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(self.n_feature_maps * 8),
            nn.ReLU(True),
            U.spectral_norm(
                nn.ConvTranspose2d(
                    self.n_feature_maps * 8,
                    self.n_feature_maps * 4,
                    4,
                    2,
                    1,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(self.n_feature_maps * 4),
            nn.ReLU(True),
            U.spectral_norm(
                nn.ConvTranspose2d(
                    self.n_feature_maps * 4,
                    self.n_feature_maps * 2,
                    4,
                    2,
                    1,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(self.n_feature_maps * 2),
            nn.ReLU(True),
            U.spectral_norm(
                nn.ConvTranspose2d(
                    self.n_feature_maps * 2, self.n_feature_maps, 4, 2, 1, bias=False
                )
            ),
            nn.BatchNorm2d(self.n_feature_maps),
            nn.ReLU(True),
            U.spectral_norm(
                nn.ConvTranspose2d(self.n_feature_maps, 3, 4, 2, 1, bias=False)
            ),
            nn.Tanh(),
        )

    def forward(self, X, labels):
        embeds = []
        for i in range(self.n_attributes):
            embeds.append(self.embeds[0](labels[:, i]))
        aux = torch.cat(embeds, dim=1)
        X = torch.cat((X, aux.view(aux.shape[0], aux.shape[1], 1, 1)), dim=1)
        return self.conv_blocks(X)
