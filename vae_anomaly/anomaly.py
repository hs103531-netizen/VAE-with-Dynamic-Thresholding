"""재구성 오차 기반 이상 영역 검출 + 평가 지표."""
import numpy as np


def dynamic_threshold(original, reconstructed, diff_mode="signed"):
    """동적 임계값 이진화 (노트북의 create_dynamic_threshold_image).

    차이 영상(reconstructed - original)을 만들고, 영상마다 원본의 평균 밝기를
    임계값으로 써서 이진화한다.

    diff_mode:
      "signed" - 노트북과 동일. (recon - orig) > mean(orig)
                 초음파에서 병변(어두운 영역)을 VAE가 정상 조직처럼 밝게 복원하는 경우를 잡는다.
      "abs"    - |recon - orig| > mean(orig). 밝은 이상 영역까지 잡고 싶을 때.
    반환: {0,1} uint8 마스크 (H, W)
    """
    original = np.squeeze(original)
    reconstructed = np.squeeze(reconstructed)
    diff = reconstructed - original
    if diff_mode == "abs":
        diff = np.abs(diff)
    elif diff_mode != "signed":
        raise ValueError(f"알 수 없는 diff_mode: {diff_mode}")
    thr = np.mean(original)
    return (diff > thr).astype(np.uint8)


def dice_coefficient(y_true, y_pred, eps=1e-7):
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float32).ravel()
    inter = np.sum(y_true * y_pred)
    return float((2.0 * inter) / (y_true.sum() + y_pred.sum() + eps))
