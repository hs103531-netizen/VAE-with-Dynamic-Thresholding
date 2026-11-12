"""학습된 VAE로 테스트 영상을 재구성 → 동적 임계값 이진화 → (마스크가 있으면) Dice 계산.

예)
    python evaluate.py --test-dir data/Select --mask-dir data/Select_Mask
    # BUSI 원본 폴더처럼 영상과 마스크가 한 폴더에 있으면 --mask-dir 를 같은 폴더로 주면 된다.
    python evaluate.py --test-dir data/Dataset_BUSI_with_GT/benign --mask-dir data/Dataset_BUSI_with_GT/benign
"""
import argparse
import csv
import json
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import cv2
import numpy as np

from vae_anomaly.anomaly import dice_coefficient, dynamic_threshold
from vae_anomaly.data import load_images, load_masks
from vae_anomaly.model import build_vae
from vae_anomaly.visualize import save_rows


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VAE anomaly maps")
    p.add_argument("--test-dir", required=True)
    p.add_argument("--mask-dir", default=None, help="GT 마스크 폴더 (없으면 Dice 생략)")
    p.add_argument("--model-dir", default="outputs", help="train.py 의 --out-dir")
    p.add_argument("--out-dir", default=None, help="결과 저장 폴더 (기본: <model-dir>/eval)")
    p.add_argument("--diff-mode", choices=["signed", "abs"], default="signed")
    p.add_argument("--save-masks", action="store_true", help="예측 마스크를 PNG로 저장")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.model_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg = json.load(f)
    vae = build_vae(cfg["img_size"], cfg["latent_dim"])
    vae.load_weights(os.path.join(args.model_dir, "vae.weights.h5"))

    test_data, names = load_images(args.test_dir, cfg["img_size"])
    recon = vae.predict(test_data, verbose=0)
    pred_masks = np.stack([dynamic_threshold(o, r, args.diff_mode) for o, r in zip(test_data, recon)])

    rows, titles = [test_data, recon, pred_masks], ["original", "reconstruction", "pred mask"]

    if args.save_masks:
        mdir = os.path.join(out_dir, "pred_masks")
        os.makedirs(mdir, exist_ok=True)
        for n, m in zip(names, pred_masks):
            cv2.imwrite(os.path.join(mdir, os.path.splitext(n)[0] + "_pred.png"), m * 255)

    if args.mask_dir:
        gt = load_masks(args.mask_dir, names, cfg["img_size"])
        rows.append(gt); titles.append("GT mask")
        scores = [dice_coefficient(g, p) for g, p in zip(gt, pred_masks)]
        with open(os.path.join(out_dir, "dice.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "dice"])
            for n, s in zip(names, scores):
                w.writerow([n, f"{s:.4f}"])
                print(f"{n:30s} Dice: {s:.4f}")
        print(f"Mean Dice: {np.mean(scores):.4f}  (n={len(scores)})")

    save_rows(rows, titles, os.path.join(out_dir, "test_results.png"), col_titles=names)
    print(f"results saved to: {out_dir}")


if __name__ == "__main__":
    main()
