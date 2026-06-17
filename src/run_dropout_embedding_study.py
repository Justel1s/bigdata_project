import argparse
import torch
from data_loader import load_dataset
from gcn import GCN
from train import train_model


def study_dropout(dataset, data, hidden, epochs, seed):
    print("Dropout (2-layer GCN, hidden=%d):" % hidden)
    print("Dropout        Accuracy")
    print("-----------------------")
    for p in [0.0, 0.3, 0.5, 0.7]:
        torch.manual_seed(seed)
        model = GCN(dataset.num_features, hidden, dataset.num_classes,
                    num_layers=2, dropout=p)
        res = train_model(model, data, epochs=epochs)
        print(f"{p:<9} {res['best_test_acc']:.4f}")


def study_embedding_size(dataset, data, epochs, seed):
    print("Embedding size (2-layer GCN, dropout=0.5):")
    print("Hidden         Accuracy")
    print("-----------------------")
    for hidden in [8, 16, 32, 64, 128]:
        torch.manual_seed(seed)
        model = GCN(dataset.num_features, hidden, dataset.num_classes,
                    num_layers=2, dropout=0.5)
        res = train_model(model, data, epochs=epochs)
        print(f"{hidden:<9} {res['best_test_acc']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora",
                        choices=["Cora", "Citeseer", "PubMed"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset, data = load_dataset(args.dataset)
    study_dropout(dataset, data, hidden=64, epochs=args.epochs, seed=args.seed)
    study_embedding_size(dataset, data, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
