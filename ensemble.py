#!/usr/bin/env python3

import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# Import your DenseNet implementation (must be in PYTHONPATH or same folder)
import densenet

# ---------------- Dataset ----------------
class SimpleDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None, img_col=None, label_col=None):
        # read csv or excel
        if label_file.lower().endswith('.xlsx') or label_file.lower().endswith('.xls'):
            df = pd.read_excel(label_file)
        else:
            df = pd.read_csv(label_file)
        cols = list(df.columns)
        if img_col and label_col:
            fn_col, lb_col = img_col, label_col
        else:
            if 'filename' in cols and 'label' in cols:
                fn_col, lb_col = 'filename', 'label'
            else:
                fn_col, lb_col = cols[0], cols[1]

        self.items = []
        for _, row in df.iterrows():
            fname = str(row[fn_col])
            path = os.path.join(img_dir, fname)
            if not os.path.exists(path):
                # warn and skip
                print(f"Warning: missing test image {path}, skipping")
                continue
            lb = int(row[lb_col])
            self.items.append((path, lb))
        self.transform = transform
        print(f"Test dataset: {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, lb = self.items[idx]
        img = Image.open(p).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, lb

# ---------------- Helpers ----------------
def infer_group_name(model_name):
    """
    Convert 'densenet_fold1' or 'densenet_plane_DL_1' etc into group prefix.
    Strategy: strip trailing '_fold\d+' or '_fold\d+' patterns, else strip trailing fold digits.
    """
    m = re.match(r'(.+)_fold\d+$', model_name)
    if m:
        return m.group(1)
    # try other common patterns: trailing digits
    m2 = re.match(r'(.+)_\d+$', model_name)
    if m2:
        return m2.group(1)
    return model_name

def build_model(args, device, n_classes):
    net = densenet.DenseNet(growthRate=args.growth_rate,
                            depth=args.depth,
                            reduction=args.reduction,
                            bottleneck=args.bottleneck,
                            nClasses=n_classes)
    net = net.to(device)
    return net

def load_checkpoint_to_model(net, ckpt_path, device):
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    state = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in state:
        sd = state['state_dict']
    else:
        sd = state
    net.load_state_dict(sd)
    net.eval()

# ---------------- Main ensemble logic ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv-csv', type=str, required=True, help='results/cv_metrics.csv from train.py')
    parser.add_argument('--test-img-dir', type=str, required=True)
    parser.add_argument('--test-label-file', type=str, required=True)
    parser.add_argument('--img-col', type=str, default=None)
    parser.add_argument('--label-col', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save-dir', type=str, default='results')
    # DenseNet params (must match those used in training)
    parser.add_argument('--growth-rate', type=int, default=12)
    parser.add_argument('--depth', type=int, default=100)
    parser.add_argument('--reduction', type=float, default=0.5)
    parser.add_argument('--bottleneck', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
    ensure_dir(args.save_dir)

    # read cv metrics
    df = pd.read_csv(args.cv_csv)
    required_cols = {'model','fold','val_accuracy','ckpt_path'}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"cv_metrics.csv must contain columns: {required_cols}. Got: {df.columns}")

    # group by inferred model prefix
    groups = defaultdict(list)  # prefix -> list of rows
    for _, row in df.iterrows():
        model_name = str(row['model'])
        prefix = infer_group_name(model_name)
        groups[prefix].append(row)

    # compute avg val_accuracy per group and choose best ckpt per group
    group_avg = {}
    group_best_ckpt = {}
    for prefix, rows in groups.items():
        accs = [float(r['val_accuracy']) for r in rows]
        avg = float(np.mean(accs))
        group_avg[prefix] = avg
        # pick ckpt with highest val_accuracy among rows
        best_row = max(rows, key=lambda r: float(r['val_accuracy']))
        ckpt_path = best_row['ckpt_path']
        group_best_ckpt[prefix] = ckpt_path

    # normalize weights
    total = sum(group_avg.values())
    if total == 0:
        raise ValueError("Sum of group accuracies is zero. Check cv_metrics.csv values.")
    weights = {g: (group_avg[g] / total) for g in group_avg}

    print("Ensemble groups and weights:")
    for g in weights:
        print(f"  {g}: avg_acc={group_avg[g]:.4f}, weight={weights[g]:.4f}, best_ckpt={group_best_ckpt[g]}")

    # build test dataset & loader (use same val transform as train.py)
    val_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_ds = SimpleDataset(args.test_img_dir, args.test_label_file, transform=val_transform,
                            img_col=args.img_col, label_col=args.label_col)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type=='cuda'))

    # Determine number of classes from dataset labels
    all_labels = [lb for _, lb in test_ds.items]
    if len(all_labels) == 0:
        raise RuntimeError("No test samples found.")
    n_classes = len(set(all_labels))
    print(f"Test set classes detected: {n_classes}")

    # For each group, load model and compute probs on test set
    group_probs = {}  # prefix -> ndarray (N, C)
    for prefix, ckpt in group_best_ckpt.items():
        print(f"Loading model for group {prefix} from {ckpt}")
        net = build_model(args, device, n_classes)
        load_checkpoint_to_model(net, ckpt, device)
        probs_list = []
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                out = net(imgs)  # logits
                probs = F.softmax(out, dim=1).cpu().numpy()
                probs_list.append(probs)
        probs_all = np.vstack(probs_list)  # shape (N, C)
        if probs_all.shape[0] != len(test_ds):
            # safety check (should match)
            print(f"Warning: probs length {probs_all.shape[0]} != test samples {len(test_ds)}")
        group_probs[prefix] = probs_all

    # Weighted ensemble
    # ensure same order of samples across groups (DataLoader preserves order)
    sample_count = len(test_ds)
    C = n_classes
    ensemble_probs = np.zeros((sample_count, C), dtype=float)
    for g, probs in group_probs.items():
        w = weights[g]
        ensemble_probs += w * probs

    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    test_targets = np.array([lb for _, lb in test_ds.items])

    acc = accuracy_score(test_targets, ensemble_preds)
    cm = confusion_matrix(test_targets, ensemble_preds)
    auc = None
    if C == 2:
        try:
            auc = roc_auc_score(test_targets, ensemble_probs[:,1])
        except Exception:
            auc = None

    print(f"Ensemble accuracy: {acc:.4f}")
    if auc is not None:
        print(f"Ensemble AUC: {auc:.4f}")
    print("Confusion matrix:")
    print(cm)

    # save results
    probs_path = os.path.join(args.save_dir, 'ensemble_probs.npy')
    preds_path = os.path.join(args.save_dir, 'ensemble_preds.npy')
    np.save(probs_path, ensemble_probs)
    np.save(preds_path, ensemble_preds)
    print(f"Saved ensemble probabilities to {probs_path}")
    print(f"Saved ensemble preds to {preds_path}")

    # save metrics json
    metrics = {
        'accuracy': float(acc),
        'auc': float(auc) if auc is not None else None,
        'confusion_matrix': cm.tolist(),
        'weights': weights,
        'groups': group_best_ckpt
    }
    metrics_path = os.path.join(args.save_dir, 'ensemble_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved ensemble metrics to {metrics_path}")

if __name__ == "__main__":
    main()
