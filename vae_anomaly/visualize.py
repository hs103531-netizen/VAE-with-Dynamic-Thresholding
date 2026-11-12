"""시각화 (plt.show 대신 파일로 저장 — VSCode/터미널에서도 동작)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_rows(rows, row_titles, path, n=10, col_titles=None):
    """rows: [배열1, 배열2, ...] 각 (N, H, W[,1]). 앞에서 n장을 행별로 그려 저장."""
    n = min(n, len(rows[0]))
    fig, axes = plt.subplots(len(rows), n, figsize=(2.2 * n, 2.3 * len(rows)), squeeze=False)
    for r, (arr, title) in enumerate(zip(rows, row_titles)):
        for c in range(n):
            ax = axes[r][c]
            ax.imshow(np.squeeze(arr[c]), cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(title, fontsize=10)
            if r == 0 and col_titles is not None:
                ax.set_title(col_titles[c], fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_loss_curve(history, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["loss"], label="train")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("loss"); ax.legend()
    ax.set_title("VAE loss")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
