import itertools
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from single_file import KeGCNv2, KeGATv2
# Hyperparameter grid
from torch_geometric.datasets import Planetoid


# Updated load_data function
def load_data():
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0]
    data.num_classes = data.y.max().item() + 1  # Add num_classes attribute
    return data

# Updated load_model function remains the same
def load_model(model_name, num_features, hidden_dim, num_classes):
    if model_name == 'KeGCN':
        return KeGCNv2(num_features, hidden_dim, num_classes)
    elif model_name == 'KeGAT':
        return KeGATv2(num_features, hidden_dim, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# Learning rate finder
def lr_find(model, data, beta, lr_range=(1e-5, 1), num_iters=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_range[0], weight_decay=5e-4)
    lr_lambda = lambda x: lr_range[0] * (lr_range[1] / lr_range[0]) ** (x / num_iters)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    losses = []
    lrs = []

    for _ in range(num_iters):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask]) + beta * (model.enhance.rule_net[0].weight**2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())

    return lrs, losses


param_grid = {
    'lr': [0.001, 0.005, 0.01],
    'beta': [0.1, 0.5, 1.0],
    'num_rules': [2, 3, 4],
    'model': ['KeGCN', 'KeGAT']
}

# Store results
results = []
data = load_data()


# Grid search
for lr, beta, num_rules, model_name in itertools.product(*param_grid.values()):
    accuracies = []
    train_accuracies = []
    val_accuracies = []

    for _ in range(3):  # 3 runs per configuration
        model = load_model(model_name, data.num_features, 16, data.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        # Training loop 
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask]) + beta * (model.enhance.rule_net[0].weight**2).mean()
            loss.backward()
            optimizer.step()

            # Evaluate train/validation accuracy per epoch
            model.eval()
            _, pred = out.max(dim=1)
            train_acc = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
            val_acc = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

        # Evaluate test accuracy
        model.eval()
        _, pred = model(data.x, data.edge_index).max(dim=1)
        test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        accuracies.append(test_acc)

    # Learning rate finder
    model = load_model(model_name, data.num_features, 16, data.num_classes)
    lrs, losses = lr_find(model, data, beta)

    # Plot learning rate finder
    plt.figure()
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title(f'LR Finder for {model_name} (beta={beta}, num_rules={num_rules})')
    plt.savefig(f'lr_finder_{model_name}_beta{beta}_rules{num_rules}.png')
    plt.close()

    # Store results
    results.append({
        'model': model_name,
        'lr': lr,
        'beta': beta,
        'num_rules': num_rules,
        'mean_acc': torch.tensor(accuracies).mean().item(),
        'std_acc': torch.tensor(accuracies).std().item(),
        'train_acc': torch.tensor(train_accuracies).mean().item(),
        'val_acc': torch.tensor(val_accuracies).mean().item()
    })

# Create and save results table
df = pd.DataFrame(results).sort_values('mean_acc', ascending=False)
print(df.head())
df.to_csv('hyperparameter_results.csv', index=False)

# Plot train vs validation performance
plt.figure(figsize=(10, 6))
for result in results:
    plt.plot(result['train_acc'], label=f"Train ({result['model']})")
    plt.plot(result['val_acc'], label=f"Validation ({result['model']})")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')
plt.legend()
plt.grid()
plt.savefig('train_vs_validation_accuracy.png')
plt.show()
