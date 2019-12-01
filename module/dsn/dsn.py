import torch.nn as nn
import torchvision.models as models

from module.dsn.utils import Flatten
from module.da.utils import GradientReversalLayer


class DSN(nn.Module):
    def __init__(self, feature_dim=64, latent_dim=512, p=0.5):
        super(DSN, self).__init__()

        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.p = p

        backboneA = models.resnet18(pretrained=True)
        self.source_encoder = nn.Sequential(
            *(list(backboneA.children())[:-3]),
            Flatten(),
            nn.Linear(1024, self.latent_dim),
            nn.ReLU(True),
        )

        backboneB = models.resnet18(pretrained=True)
        self.target_encoder = nn.Sequential(
            *(list(backboneB.children())[:-3]),
            Flatten(),
            nn.Linear(1024, self.latent_dim),
            nn.ReLU(True),
        )

        backboneC = models.resnet18(pretrained=True)
        self.shared_encoder = nn.Sequential(
            *(list(backboneC.children())[:-3]),
            Flatten(),
            nn.Linear(1024, self.latent_dim),
            nn.ReLU(True),
        )

        # decoders
        self.shared_decoder_linear = nn.Sequential(
            nn.Linear(self.latent_dim, self.feature_dim * 2 * 10 * 10),
            nn.BatchNorm1d(self.feature_dim * 2 * 10 * 10),
            nn.ReLU(True),
        )
        self.shared_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.feature_dim * 2, self.feature_dim, 4, 2, 1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.feature_dim, self.feature_dim, 4),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.feature_dim, self.feature_dim, 4),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(True),
            nn.Dropout2d(self.p),
            nn.ConvTranspose2d(self.feature_dim, 3, 3),
        )

        # classifiers
        self.shared_encoder_label_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(self.p),
            nn.Linear(1024, 10),
            nn.LogSoftmax(dim=1),
        )

        self.shared_encoder_domain_classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, X, Lambda, source=False):

        if source:
            private_latent = self.source_encoder(X)
        else:
            private_latent = self.target_encoder(X)

        shared_latent = self.shared_encoder(X)
        label = self.shared_encoder_label_predictor(shared_latent)
        reverse_features = GradientReversalLayer.apply(shared_latent, Lambda)
        domain = self.shared_encoder_domain_classifier(reverse_features)

        recon = private_latent + shared_latent
        recon = self.shared_decoder_linear(recon)
        recon = recon.view(-1, self.feature_dim * 2, 10, 10)
        recon = self.shared_decoder(recon)

        return label, domain, shared_latent, private_latent, recon

    def feature_extractor(self, X):
        features = self.shared_encoder(X)

        return features
