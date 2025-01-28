import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from single_file import KeGCNv2, KeGATv2
from torch_geometric.datasets import Planetoid
# Hyperparameter grid
param_grid = {
    'lr': [0.001, 0.005, 0.01],
    'num_rules': [2, 3, 4]
}

# Updated load_model function
def load_model(model_name, num_features, hidden_dim, num_classes):
    if model_name == 'KeGCN':
        return KeGCNv2(num_features, hidden_dim, num_classes)
    elif model_name == 'KeGAT':
        return KeGATv2(num_features, hidden_dim, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# Training and evaluation function
def train_and_evaluate(model, data, lr, num_epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        _, pred = out.max(dim=1)

        train_correct = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        train_acc = train_correct / data.train_mask.sum().item()
        train_acc_history.append(train_acc)

        val_correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        val_acc = val_correct / data.test_mask.sum().item()
        val_acc_history.append(val_acc)

    return train_acc_history, val_acc_history


# Updated load_data function
def load_data():
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0]
    data.num_classes = data.y.max().item() + 1  # Add num_classes attribute
    return data


def hyperparameter_tuning():
    dataset = load_data()
    data = dataset
    num_classes = dataset.num_classes  # Get num_classes from the dataset level
    models = ['KeGCN', 'KeGAT']
    results = {}

    # Iterate through models and hyperparameters
    for model_name in models:
        results[model_name] = {}
        for lr in param_grid['lr']:
            train_accs = []
            val_accs = []

            for num_rules in param_grid['num_rules']:
                model = load_model(model_name, data.num_features, 16, num_classes)  # Use num_classes
                train_acc_history, val_acc_history = train_and_evaluate(model, data, lr)
                train_accs.append(train_acc_history)
                val_accs.append(val_acc_history)

            # Average train/val accuracy across `num_rules`
            avg_train_acc = np.mean(train_accs, axis=0)
            avg_val_acc = np.mean(val_accs, axis=0)
            results[model_name][lr] = (avg_train_acc, avg_val_acc)

    # Plot results
    for model_name, lr_results in results.items():
        plt.figure(figsize=(10, 6))
        for lr, (train_acc, val_acc) in lr_results.items():
            plt.plot(val_acc, label=f'LR={lr}')

        plt.title(f'{model_name}: Validation Accuracy vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{model_name}_accuracy_vs_epoch.png')
        plt.show()

if __name__ == "__main__":
    hyperparameter_tuning()
