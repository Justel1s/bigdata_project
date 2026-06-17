import torch
import torch.nn.functional as F


def accuracy(logits, labels, mask):
    preds = logits[mask].argmax(dim=1)
    correct = (preds == labels[mask]).sum().item()
    return correct / int(mask.sum())


def train_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4, verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_test_acc = 0.0
    history = {"loss": [], "val_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_acc = accuracy(logits, data.y, data.val_mask)
            test_acc = accuracy(logits, data.y, data.test_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        history["loss"].append(loss.item())
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | loss {loss.item():.4f} | "
                  f"val {val_acc:.4f} | test {test_acc:.4f}")

    return {
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "history": history,
    }
