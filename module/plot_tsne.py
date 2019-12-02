import os
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from module.utils import set_random_seed
from module.dataset.digit import DigitDataset
from module.da.dann import DANN
from module.dsn.dsn import DSN

set_random_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_tsne(
    source_features_ls,
    source_labels_ls,
    source_domains_ls,
    target_features_ls,
    target_labels_ls,
    target_domains_ls,
    fname,
    sample_size=250,
):
    source_features_ls = np.concatenate(source_features_ls)
    source_labels_ls = np.concatenate(source_labels_ls)
    source_domains_ls = np.concatenate(source_domains_ls)

    target_features_ls = np.concatenate(target_features_ls)
    target_labels_ls = np.concatenate(target_labels_ls)
    target_domains_ls = np.concatenate(target_domains_ls)

    X = []
    X_labels = []
    X_domains = []

    for i in range(10):
        source_select_idxs = np.where(source_labels_ls == i)[0]
        target_select_idxs = np.where(target_labels_ls == i)[0]

        print(source_select_idxs.shape, target_select_idxs.shape)

        source_sample = random.sample(source_select_idxs.tolist(), sample_size)
        target_sample = random.sample(target_select_idxs.tolist(), sample_size)

        X.append(source_features_ls[source_sample])
        X.append(target_features_ls[target_sample])

        size = len(source_sample) + len(target_sample)
        X_labels.append([i] * size)

        X_domains.append([0] * len(source_sample))
        X_domains.append([1] * len(target_sample))

    X = np.concatenate(X)
    X_labels = np.concatenate(X_labels)
    X_domains = np.concatenate(X_domains)

    tsne = TSNE(n_components=2, verbose=1)
    X_embed = tsne.fit_transform(X)

    plt.clf()
    plt.figure(figsize=(8, 6))
    for i in range(10):
        select_idxs = np.where(X_labels == i)[0]
        select_domain_idxs = np.where(X_domains[select_idxs] == 1)[0]
        plt.scatter(
            x=X_embed[select_idxs[select_domain_idxs], 0],
            y=X_embed[select_idxs[select_domain_idxs], 1],
            c="C{}".format(i),
        )

    for i in range(10):
        select_idxs = np.where(X_labels == i)[0]
        select_domain_idxs = np.where(X_domains[select_idxs] == 0)[0]
        plt.scatter(
            x=X_embed[select_idxs[select_domain_idxs], 0],
            y=X_embed[select_idxs[select_domain_idxs], 1],
            c="C{}".format(i),
            label=i,
        )
    plt.legend()
    plt.savefig(os.path.join(fname, "label.png"))
    plt.close()

    plt.clf()
    plt.figure(figsize=(8, 6))
    for i in range(2):
        select_idxs = np.where(X_domains == i)[0]
        plt.scatter(
            x=X_embed[select_idxs, 0],
            y=X_embed[select_idxs, 1],
            c="C0" if i == 0 else "C3",
            label="source" if i == 0 else "target",
        )
    plt.legend()
    plt.savefig(os.path.join(fname, "domain.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot latent features into 2D t-SNE graph.")

    parser.add_argument("data_dir", type=str, help="Directory storing data (mnistm and svhn).")
    parser.add_argument("source_domain", type=str, help="Source domain.")
    parser.add_argument("output_image", type=str, help="Output image name.")
    parser.add_argument("--sample_size", type=int, default=50, help="How many sample per class.")

    args = parser.parse_args()

    if args.source_domain == "svhn":
        dataA_datapath = os.path.join(args.data_dir, "svhn", "test.csv")
        dataA_imagedir = os.path.join(args.data_dir, "svhn", "test")
        dataB_datapath = os.path.join(args.data_dir, "mnistm", "test.csv")
        dataB_imagedir = os.path.join(args.data_dir, "mnistm", "test")

        model_dann_path = os.path.join("models", "dann_s2m", "model_best.pth.tar")
        model_dann = DANN()
        model_dann.load_state_dict(torch.load(model_dann_path))
        model_dann = model_dann.to(device)
        model_dann.eval()

        model_dsn_path = os.path.join("models", "dsn_s2m", "model_best.pth.tar")
        model_dsn = DSN()
        model_dsn.load_state_dict(torch.load(model_dsn_path))
        model_dsn = model_dsn.to(device)
        model_dsn.eval()

    else:
        dataA_datapath = os.path.join(args.data_dir, "mnistm", "test.csv")
        dataA_imagedir = os.path.join(args.data_dir, "mnistm", "test")
        dataB_datapath = os.path.join(args.data_dir, "svhn", "test.csv")
        dataB_imagedir = os.path.join(args.data_dir, "svhn", "test")

        model_dann_path = os.path.join("models", "dann_m2s", "model_best.pth.tar")
        model_dann = DANN()
        model_dann.load_state_dict(torch.load(model_dann_path))
        model_dann = model_dann.to(device)
        model_dann.eval()

        model_dsn_path = os.path.join("models", "dns_m2s", "model_best.pth.tar")
        model_dsn = DSN()
        model_dsn.load_state_dict(torch.load(model_dsn_path))
        model_dsn = model_dsn.to(device)
        model_dsn.eval()

    source_dataset = DigitDataset(dataA_datapath, dataA_imagedir)
    target_dataset = DigitDataset(dataB_datapath, dataB_imagedir)

    with torch.no_grad():
        # dann
        dann_source_features_ls = []
        dann_source_labels_ls = []
        dann_source_domains_ls = []

        dann_target_features_ls = []
        dann_target_labels_ls = []
        dann_target_domains_ls = []

        # dsn
        dsn_source_features_ls = []
        dsn_source_labels_ls = []
        dsn_source_domains_ls = []

        dsn_target_features_ls = []
        dsn_target_labels_ls = []
        dsn_target_domains_ls = []

        dataloader = DataLoader(source_dataset, batch_size=128, shuffle=False)
        for idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            select_idxs = list(range(labels.shape[0]))

            # dann
            label_preds, _ = model_dann(images, 1.0)
            label_preds = torch.exp(label_preds).max(dim=1)[1]

            features = F.relu(model_dann.feature_extractor(images))
            features = features[select_idxs]
            bs = features.shape[0]
            dann_source_features_ls.append(features.view(bs, -1).detach().cpu().numpy())
            dann_source_labels_ls.append(labels[select_idxs].detach().cpu().numpy())
            dann_source_domains_ls.append([0] * bs)

            # dsn
            label_preds, _, _, _, _ = model_dsn(images, 1.0)
            label_preds = torch.exp(label_preds).max(dim=1)[1]

            features = F.relu(model_dsn.feature_extractor(images))
            features = features[select_idxs]
            bs = features.shape[0]
            dsn_source_features_ls.append(features.view(bs, -1).detach().cpu().numpy())
            dsn_source_labels_ls.append(labels[select_idxs].detach().cpu().numpy())
            dsn_source_domains_ls.append([0] * bs)

        dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)
        for idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            select_idxs = list(range(labels.shape[0]))

            # dann
            label_preds, _ = model_dann(images, 1.0)
            label_preds = torch.exp(label_preds).max(dim=1)[1]

            features = F.relu(model_dann.feature_extractor(images))
            features = features[select_idxs]
            bs = features.shape[0]
            dann_target_features_ls.append(features.view(bs, -1).detach().cpu().numpy())
            dann_target_labels_ls.append(labels[select_idxs].detach().cpu().numpy())
            dann_target_domains_ls.append([0] * bs)

            # dsn
            label_preds, _, _, _, _ = model_dsn(images, 1.0)
            label_preds = torch.exp(label_preds).max(dim=1)[1]

            features = F.relu(model_dsn.feature_extractor(images))
            features = features[select_idxs]
            bs = features.shape[0]
            dsn_target_features_ls.append(features.view(bs, -1).detach().cpu().numpy())
            dsn_target_labels_ls.append(labels[select_idxs].detach().cpu().numpy())
            dsn_target_domains_ls.append([0] * bs)

    plot_tsne(
        dann_source_features_ls,
        dann_source_labels_ls,
        dann_source_domains_ls,
        dann_target_features_ls,
        dann_target_labels_ls,
        dann_target_domains_ls,
        os.path.join(args.output_image, "dann"),
        sample_size=args.sample_size,
    )

    plot_tsne(
        dsn_source_features_ls,
        dsn_source_labels_ls,
        dsn_source_domains_ls,
        dsn_target_features_ls,
        dsn_target_labels_ls,
        dsn_target_domains_ls,
        os.path.join(args.output_image, "dsn"),
        sample_size=args.sample_size,
    )
