from torch_geometric.datasets import Planetoid

# The three datasets the project asks for.
AVAILABLE_DATASETS = ["Cora", "Citeseer", "PubMed"]


def load_dataset(name="Cora", root="data"):
    if name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose one of {AVAILABLE_DATASETS}."
        )

    dataset = Planetoid(root=f"{root}/{name}", name=name)
    data = dataset[0]
    return dataset, data


if __name__ == "__main__":
    for ds_name in AVAILABLE_DATASETS:
        dataset, data = load_dataset(ds_name)
        print(f"\n{ds_name} -------------")
        print(f"Nodes:    {data.num_nodes}")
        print(f"Edges:    {data.num_edges}")
        print(f"Features: {dataset.num_features}")
        print(f"Classes:  {dataset.num_classes}")
        print(f"Train/validation/test: "
              f"{int(data.train_mask.sum())} / "
              f"{int(data.val_mask.sum())} / "
              f"{int(data.test_mask.sum())}")
