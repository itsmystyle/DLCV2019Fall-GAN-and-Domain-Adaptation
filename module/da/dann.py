import torch.nn as nn

from module.da.utils import GradientReversalLayer


class DANN(nn.Module):
    def __init__(self, n_features=64, p=0.1):
        super(DANN, self).__init__()

        self.n_features = n_features
        self.p = p

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, self.n_features, 5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, 5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(self.p),
            nn.MaxPool2d(2),
            nn.ReLU(True),
        )

        self.label_predictor = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
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
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

    def forward(self, X, Lambda):
        features = self.feature_extractor(X)
        features = features.view(-1, 50 * 4 * 4)

        # label classification
        label_output = self.label_predictor(features)

        # domain classification
        reverse_features = GradientReversalLayer.apply(features, Lambda)
        domain_output = self.domain_classifier(reverse_features)

        return label_output, domain_output

    def extract_feature(self, X):
        features = self.feature_extractor(X)
        features = features.view(-1, 50 * 4 * 4)

        return features
