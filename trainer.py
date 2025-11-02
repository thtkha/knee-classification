from sympy.logic.inference import valid

from args import get_args
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os

def train_model(model, train_loader, val_loader, fold=0):
    args = get_args()

    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_balanced_accs = []
    val_balanced_accs = []
    train_roc_aucs = []
    val_roc_aucs = []
    train_avg_precisions = []
    val_avg_precisions = []

    best_balanced_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        training_loss = 0
        train_preds = []
        train_targets = []
        train_probs = []

        for batch in train_loader:
            inputs = batch['img']
            targets = batch['label']

            # reset the gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
            train_probs.extend(probs.detach().cpu().numpy())

        # Calculate training metrics
        avg_train_loss = training_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_balanced_acc = balanced_accuracy_score(train_targets, train_preds)
        train_balanced_accs.append(train_balanced_acc)

        # Calculate ROC-AUC and Average Precision for multiclass (one-vs-rest)
        train_targets_bin = label_binarize(train_targets, classes=[0, 1, 2, 3, 4])
        train_probs_array = np.array(train_probs)
        
        train_roc_auc = roc_auc_score(train_targets_bin, train_probs_array, average='macro', multi_class='ovr')
        train_roc_aucs.append(train_roc_auc)
        
        train_avg_precision = average_precision_score(train_targets_bin, train_probs_array, average='macro')
        train_avg_precisions.append(train_avg_precision)

        # Validation phase
        val_loss, val_balanced_acc, val_roc_auc, val_avg_precision = validate_model(
            model, val_loader, criterion
        )

        val_losses.append(val_loss)
        val_balanced_accs.append(val_balanced_acc)
        val_roc_aucs.append(val_roc_auc)
        val_avg_precisions.append(val_avg_precision)

        print(f'Epoch-{epoch + 1}/{args.epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Balanced Acc: {train_balanced_acc:.4f}, '
              f'Train ROC-AUC: {train_roc_auc:.4f}, Train Avg Precision: {train_avg_precision:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}, '
              f'Val ROC-AUC: {val_roc_auc:.4f}, Val Avg Precision: {val_avg_precision:.4f}')

        # Save the best model based on validation balanced accuracy
        if val_balanced_acc > best_balanced_acc:
            best_balanced_acc = val_balanced_acc
            best_epoch = epoch + 1
            best_model_path = os.path.join(args.out_dir, f'best_model_fold_{fold}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'balanced_accuracy': best_balanced_acc,
                'roc_auc': val_roc_auc,
                'avg_precision': val_avg_precision,
            }, best_model_path)
            print(f'  -> Best model saved with Balanced Accuracy: {best_balanced_acc:.4f}')

    print(f'\nTraining completed. Best Balanced Accuracy: {best_balanced_acc:.4f} at epoch {best_epoch}')

    # Plot and save the metrics
    plot_metrics(train_losses, val_losses, train_balanced_accs, val_balanced_accs,
                 train_roc_aucs, val_roc_aucs, train_avg_precisions, val_avg_precisions,
                 fold, args.out_dir)


def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []
    val_probs = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['img']
            targets = batch['label']

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()

            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    avg_val_loss = val_loss / len(val_loader)
    val_balanced_acc = balanced_accuracy_score(val_targets, val_preds)

    # Calculate ROC-AUC and Average Precision for multiclass
    val_targets_bin = label_binarize(val_targets, classes=[0, 1, 2, 3, 4])
    val_probs_array = np.array(val_probs)
    
    val_roc_auc = roc_auc_score(val_targets_bin, val_probs_array, average='macro', multi_class='ovr')
    val_avg_precision = average_precision_score(val_targets_bin, val_probs_array, average='macro')

    return avg_val_loss, val_balanced_acc, val_roc_auc, val_avg_precision


def plot_metrics(train_losses, val_losses, train_balanced_accs, val_balanced_accs,
                 train_roc_aucs, val_roc_aucs, train_avg_precisions, val_avg_precisions,
                 fold, out_dir):
    """Plot and save training and validation metrics"""
    
    epochs = range(1, len(train_losses) + 1)
    
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss over Epochs', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Balanced Accuracy
    axes[0, 1].plot(epochs, train_balanced_accs, 'b-', label='Training Balanced Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, val_balanced_accs, 'r-', label='Validation Balanced Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Balanced Accuracy', fontsize=12)
    axes[0, 1].set_title('Balanced Accuracy over Epochs', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: ROC-AUC
    axes[1, 0].plot(epochs, train_roc_aucs, 'b-', label='Training ROC-AUC', linewidth=2)
    axes[1, 0].plot(epochs, val_roc_aucs, 'r-', label='Validation ROC-AUC', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('ROC-AUC Score', fontsize=12)
    axes[1, 0].set_title('ROC-AUC over Epochs', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Average Precision
    axes[1, 1].plot(epochs, train_avg_precisions, 'b-', label='Training Avg Precision', linewidth=2)
    axes[1, 1].plot(epochs, val_avg_precisions, 'r-', label='Validation Avg Precision', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Average Precision', fontsize=12)
    axes[1, 1].set_title('Average Precision over Epochs', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(out_dir, f'training_metrics_fold_{fold}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'\nMetrics plot saved to: {plot_path}')
    plt.close()