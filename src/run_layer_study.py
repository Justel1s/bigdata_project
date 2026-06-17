import argparse
import torch

from data_loader import load_dataset
from gcn import GCN
from train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora",
                        choices=["Cora", "Citeseer", "PubMed"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset, data = load_dataset(args.dataset)
    depths = [1, 2, 3, 4, 6, 8]

    print(f"\nOversmoothing (GCN)\n")
    print("Layers        Accuracy")
    print("----------------------")
    for num_layers in depths:
        torch.manual_seed(args.seed)
        model = GCN(dataset.num_features, args.hidden, dataset.num_classes,
                    num_layers=num_layers, dropout=0.5)
        res = train_model(model, data, epochs=args.epochs)
        print(f"{num_layers:<8} {res['best_test_acc']:.4f}")


if __name__ == "__main__":
    main()
