"""정상(normal) 영상만으로 VAE를 학습한다.

예)
    python train.py --train-dir data/Dataset_BUSI_with_GT/normal --epochs 500
"""
import argparse
import json
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np

from vae_anomaly.data import load_images
from vae_anomaly.model import build_vae
from vae_anomaly.visualize import save_loss_curve, save_rows


def parse_args():
    p = argparse.ArgumentParser(description="Train VAE on normal images")
    p.add_argument("--train-dir", required=True, help="정상 영상 폴더 (예: BUSI/normal)")
    p.add_argument("--out-dir", default="outputs", help="가중치/그림 저장 폴더")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.0,
                   help="검증 비율 (0이면 노트북처럼 검증 없이 학습)")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    keras.utils.set_random_seed(args.seed)

    train_data, names = load_images(args.train_dir, args.img_size)
    print(f"train images: {train_data.shape}  (from {args.train_dir})")

    vae = build_vae(args.img_size, args.latent_dim)
    vae.encoder.summary()
    vae.decoder.summary()
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr))

    if args.val_split > 0:
        idx = np.random.permutation(len(train_data))
        n_val = max(1, int(len(train_data) * args.val_split))
        val, tr = train_data[idx[:n_val]], train_data[idx[n_val:]]
        hist = vae.fit(tr, validation_data=(val,), epochs=args.epochs, batch_size=args.batch_size)
    else:
        hist = vae.fit(train_data, epochs=args.epochs, batch_size=args.batch_size)

    weights_path = os.path.join(args.out_dir, "vae.weights.h5")
    vae.save_weights(weights_path)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({"img_size": args.img_size, "latent_dim": args.latent_dim}, f, indent=2)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()}, f)

    save_loss_curve(hist.history, os.path.join(args.out_dir, "loss_curve.png"))
    recon = vae.predict(train_data, batch_size=args.batch_size, verbose=0)
    save_rows([train_data, recon], ["original", "reconstruction"],
              os.path.join(args.out_dir, "train_reconstruction.png"), col_titles=names)
    print(f"saved: {weights_path}")


if __name__ == "__main__":
    main()
