import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from single_file import KeGCNv2, KeGATv2
from sklearn.metrics import precision_score, f1_score
from tqdm import tqdm, trange

# Hyperparameter grid
param_grid = {
    'lr': [0.001, 0.005, 0.01],
    'num_rules': [2, 3, 4]
}

# Load model function
def load_model(model_name, num_features, hidden_dim, num_classes):
    if model_name == 'KeGCN':
        return KeGCNv2(num_features, hidden_dim, num_classes)
    elif model_name == 'KeGAT':
        return KeGATv2(num_features, hidden_dim, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# Enhanced training and evaluation function with precision and F1 score
def train_and_evaluate(model, data, lr, num_epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    train_acc_history = []
    val_acc_history = []
    train_precision_history = []
    val_precision_history = []
    train_f1_history = []
    val_f1_history = []

    # Add progress bar for epochs
    epoch_bar = trange(num_epochs, desc=f"Training (LR={lr})", leave=False)
    
    for epoch in epoch_bar:
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        _, pred = out.max(dim=1)
        
        # Calculate metrics for training set
        train_y_true = data.y[data.train_mask].cpu().numpy()
        train_y_pred = pred[data.train_mask].cpu().numpy()
        
        train_correct = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        train_acc = train_correct / data.train_mask.sum().item()
        train_acc_history.append(train_acc)
        
        # Using sklearn for multi-class precision and F1
        train_precision = precision_score(train_y_true, train_y_pred, average='macro', zero_division=0)
        train_precision_history.append(train_precision)
        
        train_f1 = f1_score(train_y_true, train_y_pred, average='macro', zero_division=0)
        train_f1_history.append(train_f1)

        # Calculate metrics for validation/test set
        val_y_true = data.y[data.test_mask].cpu().numpy()
        val_y_pred = pred[data.test_mask].cpu().numpy()
        
        val_correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        val_acc = val_correct / data.test_mask.sum().item()
        val_acc_history.append(val_acc)
        
        val_precision = precision_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        val_precision_history.append(val_precision)
        
        val_f1 = f1_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        val_f1_history.append(val_f1)
        
        # Update progress bar with current metrics
        epoch_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'val_acc': f'{val_acc:.4f}',
            'val_f1': f'{val_f1:.4f}'
        })

    return {
        'train_acc': train_acc_history,
        'val_acc': val_acc_history,
        'train_precision': train_precision_history,
        'val_precision': val_precision_history,
        'train_f1': train_f1_history,
        'val_f1': val_f1_history
    }

# Load data function
def load_data():
    print("Loading Cora dataset...")
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0]
    data.num_classes = data.y.max().item() + 1  # Add num_classes attribute
    print(f"Dataset loaded: {len(data.x)} nodes, {data.edge_index.size(1)} edges, {data.num_classes} classes")
    return data

def hyperparameter_tuning():
    data = load_data()
    num_classes = data.num_classes
    models = ['KeGCN', 'KeGAT']
    results = {}

    # Main progress bar for models
    model_bar = tqdm(models, desc="Models", position=0)
    
    # Iterate through models and hyperparameters
    for model_name in model_bar:
        model_bar.set_description(f"Model: {model_name}")
        results[model_name] = {}
        
        # Progress bar for learning rates
        lr_bar = tqdm(param_grid['lr'], desc="Learning rates", position=1, leave=False)
        
        for lr in lr_bar:
            lr_bar.set_description(f"LR: {lr}")
            metrics_collector = {
                'train_acc': [],
                'val_acc': [],
                'train_precision': [],
                'val_precision': [],
                'train_f1': [],
                'val_f1': []
            }

            # Progress bar for num_rules
            rule_bar = tqdm(param_grid['num_rules'], desc="Num rules", position=2, leave=False)
            
            for num_rules in rule_bar:
                rule_bar.set_description(f"Rules: {num_rules}")
                model = load_model(model_name, data.num_features, 16, num_classes)
                metrics = train_and_evaluate(model, data, lr)
                
                # Collect metrics for averaging
                for key in metrics_collector:
                    metrics_collector[key].append(metrics[key])

            # Average metrics across num_rules
            avg_metrics = {}
            for key, values in metrics_collector.items():
                avg_metrics[key] = np.mean(values, axis=0)
                
            results[model_name][lr] = avg_metrics

    print("\nGenerating plots...")
    
    # Plot results for each metric
    metrics_to_plot = [
        {'name': 'val_precision', 'title': 'Validation Precision'},
        {'name': 'val_f1', 'title': 'Validation F1 Score'},
        {'name': 'val_acc', 'title': 'Validation Accuracy'}
    ]
    
    # Progress bar for model-specific plots
    plot_bar = tqdm(total=len(models) * len(metrics_to_plot) + len(param_grid['lr']) * len(metrics_to_plot), 
                    desc="Generating plots")
    
    for model_name, lr_results in results.items():
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            for lr, metrics in lr_results.items():
                plt.plot(metrics[metric['name']], label=f'LR={lr}')

            plt.title(f'{model_name}: {metric["title"]} vs. Epoch')
            plt.xlabel('Epoch')
            plt.ylabel(metric['title'])
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{model_name}_{metric["name"]}_vs_epoch.png')
            plt.close()  # Close instead of plt.show() to avoid blocking
            plot_bar.update(1)
            
    # Create comparison plots between models for each learning rate
    for lr in param_grid['lr']:
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            for model_name in models:
                plt.plot(results[model_name][lr][metric['name']], label=f'{model_name}')
                
            plt.title(f'Model Comparison: {metric["title"]} vs. Epoch (LR={lr})')
            plt.xlabel('Epoch')
            plt.ylabel(metric['title'])
            plt.legend()
            plt.grid(True)
            plt.tight_layout()  # Fixed typo from original code
            plt.savefig(f'model_comparison_{metric["name"]}_lr_{lr}_vs_epoch.png')
            plt.close()  # Close instead of plt.show() to avoid blocking
            plot_bar.update(1)
    
    plot_bar.close()
    print(f"All plots saved. Total plots generated: {len(models) * len(metrics_to_plot) + len(param_grid['lr']) * len(metrics_to_plot)}")

if __name__ == "__main__":
    hyperparameter_tuning()