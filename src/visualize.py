import argparse
import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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

    torch.manual_seed(args.seed)
    dataset, data = load_dataset(args.dataset)

    model = GCN(dataset.num_features, args.hidden, dataset.num_classes, dropout=0.5)
    train_model(model, data, epochs=args.epochs)

    model.eval()
    with torch.no_grad():
        emb = model.embed(data.x, data.edge_index).numpy()

    emb_2d = TSNE(n_components=2, init="pca", random_state=args.seed).fit_transform(emb)

    labels = data.y.numpy()
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab10", s=10)
    plt.legend(*scatter.legend_elements(), title="Class", loc="best", fontsize=8)
    plt.title(f"t-SNE with GCN embeddings")
    plt.xticks([])
    plt.yticks([])

    os.makedirs("results", exist_ok=True)
    out_path = f"results/tsne.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Finished")

if __name__ == "__main__":
    main()
