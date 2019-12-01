import torch.nn as nn
import torchvision.models as models

from module.da.utils import GradientReversalLayer
from module.dsn.utils import Flatten


class DANN(nn.Module):
    def __init__(self, n_features=64, p=0.1):
        super(DANN, self).__init__()

        self.n_features = n_features
        self.latent_dim = 512
        self.p = p

        backbone = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *(list(backbone.children())[:-3]),
            Flatten(),
            nn.Linear(1024, self.latent_dim),
            nn.ReLU(True),
        )

        self.label_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(self.p),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 10),
            nn.LogSoftmax(dim=1),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, X, Lambda):
        features = self.feature_extractor(X)

        # label classification
        label_output = self.label_predictor(features)

        # domain classification
        reverse_features = GradientReversalLayer.apply(features, Lambda)
        domain_output = self.domain_classifier(reverse_features)

        return label_output, domain_output

    def extract_feature(self, X):
        features = self.feature_extractor(X)

        return features
