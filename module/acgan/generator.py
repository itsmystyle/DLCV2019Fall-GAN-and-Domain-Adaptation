import torch
import torch.nn as nn
import torch.nn.utils as U


class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_feature_maps=64, n_class=2, embedding_dim=5):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.n_feature_maps = n_feature_maps
        self.n_class = n_class
        self.embedding_dim = embedding_dim

        self.class_embedding = nn.Embedding(self.n_class, self.embedding_dim)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d((self.latent_dim + self.embedding_dim)),
            U.spectral_norm(
                nn.ConvTranspose2d(
                    (self.latent_dim + self.embedding_dim),
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
        label_embeds = self.class_embedding(labels)
        label_embeds = label_embeds.unsqueeze(-1).unsqueeze(-1)
        X = torch.cat((X, label_embeds), dim=1)
        return self.conv_blocks(X)
