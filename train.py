#!/usr/bin/env python3
"""
import argparse
import os
import random
import shutil
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use('Agg')  # 防止弹窗
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score

# 假设 densenet.py 在同一目录中并提供 DenseNet(...) 接口
import densenet

from torch.cuda.amp import autocast, GradScaler

# -------------------- Dataset --------------------
class CustomDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None, img_col=None, label_col=None):
        """
        img_dir: 图片目录
        label_file: csv 或 excel，至少包含 filename 和 label 两列
        img_col, label_col: 可选的列名        """
        self.img_dir = img_dir
        self.transform = transform

        # 读取标签文件
        if label_file.lower().endswith('.xlsx') or label_file.lower().endswith('.xls'):
            df = pd.read_excel(label_file)
        else:
            df = pd.read_csv(label_file)

        cols = list(df.columns)
        if img_col and label_col:
            fn_col, lb_col = img_col, label_col
        else:
            # 尝试常见列名，否则取前两列
            if 'filename' in cols and 'label' in cols:
                fn_col, lb_col = 'filename', 'label'
            else:
                fn_col, lb_col = cols[0], cols[1]

        self.valid_data = []
        for _, row in df.iterrows():
            fname = str(row[fn_col])
            img_path = os.path.join(self.img_dir, fname)
            if os.path.exists(img_path):
                try:
                    label = int(row[lb_col])
                except Exception:
                    # 若 label 不能直接转型为 int，则尝试映射
                    label = row[lb_col]
                self.valid_data.append((img_path, label))
            else:
                print(f"Warning: missing file {img_path}")

        print(f"CustomDataset: found {len(self.valid_data)} valid samples in {img_dir}")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        p, lb = self.valid_data[idx]
        img = Image.open(p).convert('L')  # 灰度
        if self.transform:
            img = self.transform(img)
        return img, int(lb)


# -------------------- Utilities --------------------
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_checkpoint(state, path):
    torch.save(state, path)

def plot_and_save_auc(fpr, tpr, auc_score, path):
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'AUC={auc_score:.3f}')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# -------------------- Train / Eval --------------------
def train_one_epoch(net, device, loader, optimizer, scaler):
    net.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    n = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        with autocast():
            out = net(data)
            loss = F.cross_entropy(out, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * data.size(0)
        preds = out.detach().argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(target.cpu().numpy().tolist())
        n += data.size(0)
    avg_loss = total_loss / n if n>0 else 0.0
    acc = accuracy_score(all_targets, all_preds) if n>0 else 0.0
    return avg_loss, acc

def evaluate(net, device, loader):
    net.eval()
    total_loss = 0.0
    probs_list = []
    targets_list = []
    n = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            out = net(data)
            loss = F.cross_entropy(out, target, reduction='sum')
            total_loss += loss.item()
            probs = F.softmax(out, dim=1).cpu().numpy()
            probs_list.append(probs)
            targets_list.extend(target.cpu().numpy().tolist())
            n += data.size(0)
    if n == 0:
        return 0.0, 0.0, None, (None, None)
    all_probs = np.vstack(probs_list)
    avg_loss = total_loss / n
    preds = all_probs.argmax(axis=1)
    acc = accuracy_score(targets_list, preds)
    auc = None
    if all_probs.shape[1] == 2 and len(np.unique(targets_list)) > 1:
        try:
            auc = roc_auc_score(targets_list, all_probs[:,1])
        except Exception:
            auc = None
    return avg_loss, acc, auc, (all_probs, np.array(targets_list))


# -------------------- LR Scheduler builder --------------------
def build_scheduler(optimizer, step_size=30, gamma=0.1):
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, required=True, help='image folder')
    parser.add_argument('--label-file', type=str, required=True, help='csv/xlsx with filename,label')
    parser.add_argument('--img-col', type=str, default=None)
    parser.add_argument('--label-col', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use-aug', action='store_true', help='enable augmentation')
    parser.add_argument('--save-dir', type=str, default='results')
    parser.add_argument('--opt', choices=['sgd','adam','rmsprop'], default='sgd')
    parser.add_argument('--growth-rate', type=int, default=12)
    parser.add_argument('--depth', type=int, default=100)
    parser.add_argument('--reduction', type=float, default=0.5)
    parser.add_argument('--bottleneck', action='store_true', help='use DenseNet bottleneck/compression (BC)')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    set_seed(args.seed)
    ensure_dir(args.save_dir)

    # transforms: augmentation per paper (水平翻转、旋转、平移、亮度/对比扰动)
    if args.use_aug:
        train_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.9,1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    val_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # load dataset (用于划分索引)
    full_ds = CustomDataset(args.img_dir, args.label_file, transform=val_transform,
                            img_col=args.img_col, label_col=args.label_col)
    if len(full_ds) == 0:
        raise RuntimeError("No valid samples found. 检查 img-dir 与 label-file。")

    labels = [lb for _, lb in full_ds.valid_data]
    n_classes = len(set(labels))
    print(f"Detected classes: {n_classes}")

    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args.seed)

    scaler = GradScaler()

    cv_records = []
    fold_idx = 0
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        fold_idx = fold + 1
        print(f"\n==== Fold {fold_idx}/{k_folds} ====")

        train_items = [full_ds.valid_data[i] for i in train_idx]
        val_items = [full_ds.valid_data[i] for i in val_idx]

        class ItemsDataset(Dataset):
            def __init__(self, items, transform):
                self.items = items
                self.transform = transform
            def __len__(self):
                return len(self.items)
            def __getitem__(self, idx):
                p, lb = self.items[idx]
                img = Image.open(p).convert('L')
                if self.transform:
                    img = self.transform(img)
                return img, int(lb)

        train_dataset = ItemsDataset(train_items, transform=train_transform)
        val_dataset = ItemsDataset(val_items, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=args.cuda)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, pin_memory=args.cuda)

        # build DenseNet (DenseNet-BC if bottleneck True)
        net = densenet.DenseNet(growthRate=args.growth_rate, depth=args.depth,
                                reduction=args.reduction, bottleneck=args.bottleneck,
                                nClasses=n_classes)
        net = net.to(device)

        # optimizer per paper
        if args.opt == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        elif args.opt == 'adam':
            optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        else:
            optimizer = optim.RMSprop(net.parameters(), lr=1e-3, weight_decay=1e-4)

        scheduler = build_scheduler(optimizer, step_size=30, gamma=0.1)

        best_val_acc = 0.0
        best_ckpt = None

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(net, device, train_loader, optimizer, scaler)
            val_loss, val_acc, val_auc, (val_probs_targets) = evaluate(net, device, val_loader)
            scheduler.step()

            print(f"Fold{fold_idx} Epoch {epoch}/{args.epochs} | train_loss {train_loss:.4f} acc {train_acc:.4f} "
                  f"| val_loss {val_loss:.4f} acc {val_acc:.4f} auc {val_auc}")

            # 保存最佳模型（按 val_acc）
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_name = f"densenet_fold{fold_idx}_best.pth"
                ckpt_path = os.path.join(args.save_dir, ckpt_name)
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_acc': val_acc
                }, ckpt_path)
                best_ckpt = ckpt_path

            # 每30 epoch 保存一次 AUC 图（若为二分类并可计算）
            if val_auc is not None and epoch % 30 == 0:
                probs, targets = val_probs_targets
                try:
                    fpr, tpr, _ = roc_curve(targets, probs[:,1])
                    auc_score = roc_auc_score(targets, probs[:,1])
                    auc_path = os.path.join(args.save_dir, f"auc_fold{fold_idx}_epoch{epoch}.png")
                    plot_and_save_auc(fpr, tpr, auc_score, auc_path)
                except Exception as e:
                    print("AUC plot error:", e)

        # 记录该 fold 最佳结果，用于 ensemble 权重计算
        print(f"Fold {fold_idx} done. Best val_acc={best_val_acc:.4f}, ckpt={best_ckpt}")
        cv_records.append({
            'model': f'densenet_fold{fold_idx}',
            'fold': fold,
            'val_accuracy': float(best_val_acc),
            'ckpt_path': best_ckpt
        })

    # 保存所有 fold 的 CV 结果到 csv，ensemble 脚本将读取此文件进行加权融合
    cv_df = pd.DataFrame(cv_records)
    cv_csv = os.path.join(args.save_dir, 'cv_metrics.csv')
    cv_df.to_csv(cv_csv, index=False)
    print(f"Saved CV metrics to {cv_csv}")


if __name__ == "__main__":
    main()
