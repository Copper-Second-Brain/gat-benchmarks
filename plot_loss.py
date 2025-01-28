import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from single_file import KeGCNv2, KeGATv2
from plot_accuracy import load_data
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
def train_and_evaluate_loss(model, data, lr, num_epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_loss_history.append(loss.item())

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(val_out[data.test_mask], data.y[data.test_mask])
            val_loss_history.append(val_loss.item())

    return train_loss_history, val_loss_history

# Main function
def hyperparameter_tuning_loss():
    dataset = load_data()
    data = dataset
    num_classes = dataset.num_classes  # Get num_classes from the dataset level
    models = ['KeGCN', 'KeGAT']
    results = {}

    # Iterate through models and hyperparameters
    for model_name in models:
        results[model_name] = {}
        for lr in param_grid['lr']:
            train_losses = []
            val_losses = []

            for num_rules in param_grid['num_rules']:
                model = load_model(model_name, data.num_features, 16, num_classes)  # Use num_classes
                train_loss_history, val_loss_history = train_and_evaluate_loss(model, data, lr)
                train_losses.append(train_loss_history)
                val_losses.append(val_loss_history)

            # Average train/val loss across `num_rules`
            avg_train_loss = np.mean(train_losses, axis=0)
            avg_val_loss = np.mean(val_losses, axis=0)
            results[model_name][lr] = (avg_train_loss, avg_val_loss)

    # Plot results
    for model_name, lr_results in results.items():
        plt.figure(figsize=(10, 6))
        for lr, (train_loss, val_loss) in lr_results.items():
            plt.plot(val_loss, label=f'LR={lr}')

        plt.title(f'{model_name}: Validation Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{model_name}_loss_vs_epoch.png')
        plt.show()

if __name__ == "__main__":
    hyperparameter_tuning_loss()
