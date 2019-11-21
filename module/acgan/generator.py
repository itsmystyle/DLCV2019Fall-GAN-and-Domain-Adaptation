import torch
import torch.nn as nn
import torch.nn.utils as U


class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_feature_maps=128, n_class=2, embedding_dim=4):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.n_feature_maps = n_feature_maps
        self.n_class = n_class
        self.embedding_dim = embedding_dim

        self.bangs_embeds = nn.Embedding(self.n_class, self.embedding_dim)
        self.biglips_embeds = nn.Embedding(self.n_class, self.embedding_dim)
        self.heavymakeups_embeds = nn.Embedding(self.n_class, self.embedding_dim)
        self.highcheekbones_embeds = nn.Embedding(self.n_class, self.embedding_dim)
        self.male_embeds = nn.Embedding(self.n_class, self.embedding_dim)
        self.wearinglipstick_embeds = nn.Embedding(self.n_class, self.embedding_dim)
        self.smile_embeds = nn.Embedding(self.n_class, self.embedding_dim)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d((self.latent_dim + 7 * self.embedding_dim)),
            U.spectral_norm(
                nn.ConvTranspose2d(
                    (self.latent_dim + 7 * self.embedding_dim),
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
        bangs = self.bangs_embeds(labels[:, 0])
        biglips = self.biglips_embeds(labels[:, 1])
        hmakeups = self.heavymakeups_embeds(labels[:, 2])
        hcheekbones = self.highcheekbones_embeds(labels[:, 3])
        male = self.male_embeds(labels[:, 4])
        wlipstick = self.wearinglipstick_embeds(labels[:, 5])
        smile = self.smile_embeds(labels[:, 6])
        aux = torch.cat(
            (bangs, biglips, hmakeups, hcheekbones, male, wlipstick, smile), dim=1
        )
        X = torch.cat((X, aux.view(aux.shape[0], aux.shape[1], 1, 1)), dim=1)
        return self.conv_blocks(X)
