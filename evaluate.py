from args import get_args
import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from datasets import knee_xray_dataset
from models import MyModel
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


def load_model_for_fold(backbone, checkpoint_path, device):
    model = MyModel(backbone)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    # saved dict may contain 'model_state_dict'
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate_ensemble(models, dataloader, device):
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['img'].to(device)
            targets = batch['label'].to('cpu').numpy()

            # collect probs from each model
            probs = []
            for m in models:
                out = m(imgs)
                p = F.softmax(out, dim=1).cpu().numpy()
                probs.append(p)

            # average probabilities (soft voting)
            avg_probs = np.mean(probs, axis=0)
            preds = np.argmax(avg_probs, axis=1)

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)

    return acc, bal_acc, cm, all_targets, all_preds


def main():
    args = get_args()

    # device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

    print(f"Evaluation device: {device}")

    # load test csv
    test_csv = os.path.join(args.csv_dir, 'test.csv')
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    test_df = pd.read_csv(test_csv)

    test_dataset = knee_xray_dataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # load the 5 fold models
    models = []
    for fold in range(5):
        chk = os.path.join(args.out_dir, f'best_model_fold_{fold}.pth')
        try:
            m = load_model_for_fold(args.backbone, chk, device)
            models.append(m)
            print(f"Loaded model for fold {fold} from {chk}")
        except FileNotFoundError:
            print(f"Warning: checkpoint missing for fold {fold}: {chk} -- skipping")

    if len(models) == 0:
        raise RuntimeError("No models loaded. Ensure checkpoints exist in out_dir.")

    acc, bal_acc, cm, targets, preds = evaluate_ensemble(models, test_loader, device)

    print(f"Test Accuracy (ensemble avg probs): {acc:.4f}")
    print(f"Test Balanced Accuracy: {bal_acc:.4f}")
    print("Confusion matrix:\n", cm)

    # Optionally save per-sample predictions
    out_df = pd.DataFrame({
        'Path': test_df['Path'],
        'True': targets,
        'Pred': preds
    })
    out_path = os.path.join(args.out_dir, 'test_predictions_ensemble.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Per-sample predictions saved to: {out_path}")


if __name__ == '__main__':
    main()
