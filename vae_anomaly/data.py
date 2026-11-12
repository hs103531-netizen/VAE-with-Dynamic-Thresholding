"""이미지 / 마스크 로딩.

- 파일 목록은 항상 정렬해서 읽는다. (노트북은 os.listdir 순서를 그대로 써서
  테스트 이미지와 마스크의 순서가 어긋날 수 있었음)
- 마스크는 이미지 파일명으로 매칭한다: "benign (121).png" -> "benign (121)_mask*.png"
  BUSI에는 한 영상에 마스크가 여러 장(_mask_1.png 등)인 경우가 있어 합집합(OR)으로 합친다.
"""
import glob
import os

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder):
    """폴더 안의 영상 파일 목록 (마스크 파일 제외, 정렬)."""
    files = [
        f for f in sorted(os.listdir(folder))
        if f.lower().endswith(IMG_EXTS) and "_mask" not in f
    ]
    if not files:
        raise FileNotFoundError(f"영상 파일이 없습니다: {folder}")
    return files


def read_gray(path, img_size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"영상을 읽을 수 없습니다: {path}")
    return cv2.resize(img, (img_size, img_size))


def load_images(folder, img_size=256):
    """폴더의 영상을 (N, H, W, 1) float32 [0, 1] 배열로 읽는다. 파일명 리스트도 함께 반환."""
    names = list_images(folder)
    imgs = np.stack([read_gray(os.path.join(folder, n), img_size) for n in names])
    imgs = imgs.astype("float32")[..., None] / 255.0
    return imgs, names


def load_masks(mask_folder, image_names, img_size=256):
    """image_names 순서에 맞춰 GT 마스크를 (N, H, W) uint8 {0,1} 로 읽는다."""
    masks = []
    for name in image_names:
        stem = os.path.splitext(name)[0]
        pattern = os.path.join(glob.escape(mask_folder), glob.escape(stem) + "_mask*")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"'{name}'에 대응하는 마스크가 없습니다: {pattern}")
        m = np.zeros((img_size, img_size), dtype=np.uint8)
        for p in paths:
            m |= (read_gray(p, img_size) > 127).astype(np.uint8)
        masks.append(m)
    return np.stack(masks)
