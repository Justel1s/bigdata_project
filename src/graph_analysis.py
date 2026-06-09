from collections import Counter
import networkx as nx


def basic_stats(data):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    avg_degree = num_edges / num_nodes
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": round(avg_degree, 2),
    }


def class_distribution(data):
    labels = data.y.tolist()
    counts = Counter(labels)
    return dict(sorted(counts.items()))


def intra_inter_class_edges(data):
    src = data.edge_index[0]
    dst = data.edge_index[1]
    same_class = (data.y[src] == data.y[dst])

    intra = int(same_class.sum())
    inter = int((~same_class).sum())
    total = intra + inter
    intra_ratio = intra / total if total > 0 else 0.0

    return {
        "intra_class_edges": intra,
        "inter_class_edges": inter,
        "intra_class_ratio": round(intra_ratio, 3),
    }


def to_networkx_graph(data):
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    g.add_edges_from(data.edge_index.t().tolist())
    return g


def print_report(data, dataset_name="dataset"):
    print(f"\nData Analysis: {dataset_name}")

    stats = basic_stats(data)
    print(f"Nodes:          {stats['num_nodes']}")
    print(f"Edges:          {stats['num_edges']}")
    print(f"Average degree: {stats['avg_degree']}")

    print("\nClass distribution:")
    for cls, count in class_distribution(data).items():
        print(f"-class {cls}: {count}")

    edges = intra_inter_class_edges(data)
    print("\nEdge types:")
    print(f"-intra-class edges: {edges['intra_class_edges']}")
    print(f"-inter-class edges: {edges['inter_class_edges']}")
    print(f"-intra-class ratio: {edges['intra_class_ratio']}")


if __name__ == "__main__":
    from data_loader import load_dataset

    _, data = load_dataset("Cora")
    print_report(data, "Cora")
