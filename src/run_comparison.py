import argparse
import torch
from data_loader import load_dataset
from logreg_baseline import run_logistic_regression
from mlp_baseline import MLP
from gcn import GCN
from graph_sage import GraphSAGE
from gat import GAT
from train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora",
                        choices=["Cora", "Citeseer", "PubMed"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset, data = load_dataset(args.dataset)
    in_channels = dataset.num_features
    out_channels = dataset.num_classes

    results = {}

    results["LogReg"] = run_logistic_regression(data)["test"]

    models = {
        "MLP": MLP(in_channels, args.hidden, out_channels, dropout=args.dropout),
        "GCN": GCN(in_channels, args.hidden, out_channels, dropout=args.dropout),
        "GraphSAGE": GraphSAGE(in_channels, args.hidden, out_channels, dropout=args.dropout),
        "GAT": GAT(in_channels, args.hidden, out_channels, dropout=args.dropout),
    }

    for name, model in models.items():
        res = train_model(model, data, epochs=args.epochs)
        results[name] = res["best_test_acc"]

    print("Model                 Accuracy")
    print("------------------------------")
    for name, acc in results.items():
        print(f"{name:<16} {acc:.4f}")


if __name__ == "__main__":
    main()
