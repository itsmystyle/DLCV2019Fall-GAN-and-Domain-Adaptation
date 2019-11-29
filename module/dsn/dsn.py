import torch.nn as nn
import torch.nn.functional as F

from module.dsn.utils import Flatten
from module.da.utils import GradientReversalLayer


class DSN(nn.Module):
    def __init__(self, feature_dim=32, latent_dim=100, p=0.5):
        super(DSN, self).__init__()

        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.p = p

        def build_feature_extractor(feature_dim, latent_dim):
            return nn.Sequential(
                nn.Conv2d(3, feature_dim, 5),
                nn.BatchNorm2d(feature_dim),
                nn.MaxPool2d(2),
                nn.ReLU(True),
                nn.Conv2d(feature_dim, feature_dim * 2, 5),
                nn.BatchNorm2d(feature_dim * 2),
                nn.ReLU(True),
                nn.Dropout2d(self.p),
                nn.MaxPool2d(2),
                Flatten(),
                nn.Linear(feature_dim * 2 * 4 * 4, latent_dim),
            )

        # encoders
        self.source_encoder = build_feature_extractor(self.feature_dim, self.latent_dim)

        self.target_encoder = build_feature_extractor(self.feature_dim, self.latent_dim)

        self.shared_encoder = build_feature_extractor(self.feature_dim, self.latent_dim)

        # decoders
        self.shared_decoder_linear = nn.Sequential(
            nn.Linear(self.latent_dim, self.feature_dim * 2 * 4 * 4),
            nn.BatchNorm1d(self.feature_dim * 2 * 4 * 4),
            nn.ReLU(True),
            nn.Dropout(self.p),
        )
        self.shared_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.feature_dim * 2, self.feature_dim, 5, 2, 1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(True),
            nn.Dropout2d(self.p),
            nn.ConvTranspose2d(self.feature_dim, self.feature_dim, 5, 2, 1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(True),
            nn.Dropout2d(self.p),
            nn.ConvTranspose2d(self.feature_dim, self.feature_dim, 7),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(True),
            nn.Dropout2d(self.p),
            nn.ConvTranspose2d(self.feature_dim, 3, 4),
        )

        # classifiers
        self.shared_encoder_label_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(self.p),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1),
        )

        self.shared_encoder_domain_classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(self.p),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

    def forward(self, X, Lambda, source=False):

        if source:
            private_latent = self.source_encoder(X)
        else:
            private_latent = self.target_encoder(X)

        shared_latent = self.shared_encoder(X)
        label = self.shared_encoder_label_predictor(F.relu(shared_latent))
        reverse_features = GradientReversalLayer.apply(F.relu(shared_latent), Lambda)
        domain = self.shared_encoder_domain_classifier(reverse_features)

        recon = private_latent + shared_latent
        recon = F.relu(recon)
        recon = self.shared_decoder_linear(recon)
        recon = recon.view(-1, self.feature_dim * 2, 4, 4)
        recon = self.shared_decoder(recon)

        return label, domain, shared_latent, private_latent, recon

    def extract_feature(self, X):
        features = self.shared_encoder(X)

        return features
