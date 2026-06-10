from sklearn.linear_model import LogisticRegression


def run_logistic_regression(data):
    x = data.x.numpy()
    y = data.y.numpy()

    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x[train_mask], y[train_mask])

    return {
        "train": clf.score(x[train_mask], y[train_mask]),
        "val": clf.score(x[val_mask], y[val_mask]),
        "test": clf.score(x[test_mask], y[test_mask]),
    }


if __name__ == "__main__":
    from data_loader import load_dataset
    _, data = load_dataset("Cora")
    acc = run_logistic_regression(data)
    print(f"Logistic accuracy: {acc['test']:.4f}")
