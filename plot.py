#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('bmh')


def read_csv_auto(path):
    """
    尝试用 pandas 读取 CSV 或 TXT，返回 ndarray。
    If file missing or empty -> raise.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    # try read with pandas (it handles headers or no headers)
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 3:
        raise ValueError(f"Expect at least 3 columns in {path}, got {df.shape[1]}")
    # use first three columns
    arr = df.iloc[:, :3].to_numpy()
    return arr  # shape (N,3)


def rolling_avg(arr, window):
    """
    Compute simple moving average along axis 0 for columns 1 and 2.
    arr: shape (N,3) with columns (index, loss, err)
    returns: (idx_windowed, loss_smoothed, err_smoothed)
    If window <= 1, return raw series (no smoothing).
    """
    idx = arr[:, 0]
    loss = arr[:, 1]
    err = arr[:, 2]
    n = len(idx)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    if window <= 1 or window >= n:
        # no smoothing, return arrays trimmed to align with original idx
        return idx, loss, err
    # use convolution for moving average
    k = np.ones(window) / window
    loss_smooth = np.convolve(loss, k, mode='valid')
    err_smooth = np.convolve(err, k, mode='valid')
    idx_valid = idx[window - 1:]
    return idx_valid, loss_smooth, err_smooth


def plot_curves(exp_dir, window=392*2, out_prefix='loss'):
    train_p = os.path.join(exp_dir, 'train.csv')
    test_p = os.path.join(exp_dir, 'test.csv')

    train_arr = read_csv_auto(train_p)
    test_arr = read_csv_auto(test_p)

    # adapt window if too large
    max_win = max(1, len(train_arr) // 2)
    if window > max_win:
        window = max(1, min(window, len(train_arr)))
    idx_t, loss_t, err_t = rolling_avg(train_arr, window)
    idx_v = test_arr[:, 0]
    loss_v = test_arr[:, 1]
    err_v = test_arr[:, 2]

    # Plot 1: Loss (log-scale)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    if len(idx_t) > 0:
        ax1.plot(idx_t, loss_t, label='Train (smoothed)')
    else:
        ax1.plot(train_arr[:, 0], train_arr[:, 1], label='Train')
    ax1.plot(idx_v, loss_v, label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True)
    loss_fname = os.path.join(exp_dir, f'{out_prefix}_loss.png')
    fig1.tight_layout()
    fig1.savefig(loss_fname)
    plt.close(fig1)

    # Plot 2: Error (log-scale)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    if len(idx_t) > 0:
        ax2.plot(idx_t, err_t, label='Train (smoothed)')
    else:
        ax2.plot(train_arr[:, 0], train_arr[:, 2], label='Train')
    ax2.plot(idx_v, err_v, label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)
    err_fname = os.path.join(exp_dir, f'{out_prefix}_error.png')
    fig2.tight_layout()
    fig2.savefig(err_fname)
    plt.close(fig2)

    # Combine into one image (side-by-side) by creating a larger figure with two subplots
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    # left: loss
    if len(idx_t) > 0:
        a1.plot(idx_t, loss_t, label='Train (smoothed)')
    else:
        a1.plot(train_arr[:, 0], train_arr[:, 1], label='Train')
    a1.plot(idx_v, loss_v, label='Test')
    a1.set_xlabel('Epoch')
    a1.set_ylabel('Loss')
    a1.set_yscale('log')
    a1.legend()
    a1.grid(True)
    # right: error
    if len(idx_t) > 0:
        a2.plot(idx_t, err_t, label='Train (smoothed)')
    else:
        a2.plot(train_arr[:, 0], train_arr[:, 2], label='Train')
    a2.plot(idx_v, err_v, label='Test')
    a2.set_xlabel('Epoch')
    a2.set_ylabel('Error')
    a2.set_yscale('log')
    a2.legend()
    a2.grid(True)

    combined_fname = os.path.join(exp_dir, f'{out_prefix}-loss-error.png')
    fig.tight_layout()
    fig.savefig(combined_fname)
    plt.close(fig)

    return loss_fname, err_fname, combined_fname


def main():
    parser = argparse.ArgumentParser(description='Plot train/test loss and error curves.')
    parser.add_argument('expDir', type=str, help='experiment directory containing train.csv and test.csv')
    parser.add_argument('--window', type=int, default=392*2, help='moving average window (default 392*2)')
    parser.add_argument('--out-prefix', type=str, default='loss', help='output filename prefix')
    args = parser.parse_args()

    loss_f, err_f, comb_f = plot_curves(args.expDir, window=args.window, out_prefix=args.out_prefix)
    print(f'Created: {loss_f}')
    print(f'Created: {err_f}')
    print(f'Created: {comb_f}')


if __name__ == '__main__':
    main()
