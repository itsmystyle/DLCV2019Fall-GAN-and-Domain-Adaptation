import argparse

import torch
from torch.utils.data import DataLoader

from module.utils import set_random_seed
from module.dataset.digit import DigitDataset
from module.dsn.dsn import DSN
from module.metrics import MulticlassAccuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Adaptation predictor.")

    parser.add_argument(
        "model_dir", type=str, help="Directory path to best model stored."
    )
    parser.add_argument("test_datapath", type=str, help="Path to test dataset.")
    parser.add_argument(
        "test_images_dir", type=str, help="Path to test images directory."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed.")

    args = parser.parse_args()

    set_random_seed(args.random_seed)
    dataset = DigitDataset(args.test_datapath, args.test_images_dir)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models
    model = DSN()
    model.load_state_dict(torch.load(args.model_dir))
    model.to(device)
    model.eval()

    # Accuracy
    metric = MulticlassAccuracy()
    metric.reset()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            preds, _, _, _, _ = model(images, 1.0)
            metric.update(preds, labels)

    print("Multiclass Accuracy: {:.5f}".format(metric.get_score()))
