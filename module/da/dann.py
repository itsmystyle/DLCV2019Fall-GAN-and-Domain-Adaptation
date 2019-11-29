import torch.nn as nn
import torch.nn.functional as F

from module.da.utils import GradientReversalLayer
from module.dsn.utils import Flatten


class DANN(nn.Module):
    def __init__(self, n_features=32, p=0.1):
        super(DANN, self).__init__()

        self.n_features = n_features
        self.latent_dim = 512
        self.p = p

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, self.n_features, 5),
            nn.BatchNorm2d(self.n_features),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(self.n_features, self.n_features * 2, 5),
            nn.BatchNorm2d(self.n_features * 2),
            nn.Dropout2d(self.p),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(self.n_features * 2 * 4 * 4, self.latent_dim),
        )

        self.label_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(self.p),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

    def forward(self, X, Lambda):
        features = self.feature_extractor(X)
        features = F.relu(features)

        # label classification
        label_output = self.label_predictor(features)

        # domain classification
        reverse_features = GradientReversalLayer.apply(features, Lambda)
        domain_output = self.domain_classifier(reverse_features)

        return label_output, domain_output

    def extract_feature(self, X):
        features = self.feature_extractor(X)

        return features
