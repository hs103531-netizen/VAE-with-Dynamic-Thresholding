# 🩺 VAE with Dynamic Thresholding — 유방 초음파 병변 비지도 검출

> 정상 유방 초음파 영상만으로 VAE를 학습하고,
> 재구성 오차 + 영상별 동적 임계값으로 병변 영역을 찾는 비지도 이상 탐지 프로젝트입니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목적 | 병변 라벨 없이 정상 영상만으로 학습해 초음파 영상의 병변 영역 검출 |
| 데이터 | BUSI (Breast Ultrasound Images Dataset) |
| 모델 | Convolutional VAE (latent dim 2) |
| 핵심 아이디어 | 재구성 차이 영상을 **영상별 평균 밝기**로 이진화 (Dynamic Thresholding) |
| 평가 지표 | Dice Coefficient (GT 마스크 기준) |

---

## 📐 전체 파이프라인

```
BUSI normal 영상
        │  256×256 grayscale, 0~1 정규화
        ▼
┌─────────────────┐
│    train.py     │  Convolutional VAE 학습 (BCE + KL loss)
└────────┬────────┘
         │  vae.weights.h5
         ▼
┌─────────────────┐
│   evaluate.py   │  benign / malignant 영상 재구성
│                 │  차이 영상 = recon − original
│                 │  임계값 = mean(original) → 이진 마스크
└────────┬────────┘
         │
         ▼
   Dice Coefficient (GT 마스크 비교) · 결과 이미지 저장
```

---

## 💡 Dynamic Thresholding

- VAE는 정상 조직만 학습했기 때문에 병변 부위를 정상 조직처럼 복원하려고 함
- 초음파에서 병변은 주로 어둡게(저에코) 보이므로, 복원 영상이 원본보다 밝아지는 영역을 병변 후보로 봄
- 초음파 영상은 장비·게인 설정에 따라 밝기 편차가 커서, 고정 임계값 대신 **영상마다 원본 평균 밝기를 임계값**으로 사용

```python
diff = reconstructed - original
mask = diff > np.mean(original)
```

---

## 🗂️ 프로젝트 구조

```
VAE-with-Dynamic-Thresholding/
├── vae_anomaly/
│   ├── model.py        # Sampling 레이어, Encoder / Decoder, VAE (custom train_step)
│   ├── data.py         # 영상 / 마스크 로딩 (파일명 기준 매칭)
│   ├── anomaly.py      # Dynamic Thresholding, Dice Coefficient
│   └── visualize.py    # 결과 이미지 · loss 그래프 저장
├── train.py            # 학습 스크립트
├── evaluate.py         # 평가 스크립트
├── Variational_Auto_Encoder_Anomaly.ipynb   # Colab 실험 노트북
└── requirements.txt
```

---

## 🤖 모델 구조

| 구분 | 구성 |
|------|------|
| Encoder | Conv2D(32, stride 2) → Conv2D(64, stride 2) → Dense(16) → z_mean, z_log_var |
| Latent | 2차원, Reparameterization trick |
| Decoder | Dense(64×64×64) → Conv2DTranspose(64) → Conv2DTranspose(32) → Conv2DTranspose(1, sigmoid) |
| Loss | Binary Cross-Entropy + KL Divergence |
| Optimizer | Adam (lr 1e-3) |
| 학습 설정 | epochs 500, batch size 128 |

Keras 공식 VAE 예제(MNIST)를 기반으로 입력 크기를 256×256으로 확장했습니다.

---

## 📊 데이터

| 항목 | 내용 |
|------|------|
| 출처 | [Kaggle – Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
| 구성 | normal 133장 / benign 437장 / malignant 210장 (+ 병변 GT 마스크) |
| 학습 | normal 133장 |
| 평가 | benign · malignant 영상 + GT 마스크 |
| 전처리 | grayscale 변환, 256×256 리사이즈, 0~1 정규화 |

---

## 🛠️ 기술 스택 / 개발 환경

| 분야 | 기술 |
|------|------|
| 언어 | Python |
| 딥러닝 | TensorFlow 2.16+ / Keras 3 |
| 영상 처리 | OpenCV, NumPy |
| 시각화 | Matplotlib |
| 실험 환경 | Google Colab (T4 GPU) |
| 코드 정리 | VSCode (로컬 실행용 스크립트) |

---

## 🚀 실행 방법

### 1. 환경 설정
```bash
git clone https://github.com/hs103531-netizen/VAE-with-Dynamic-Thresholding.git
cd VAE-with-Dynamic-Thresholding
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. 데이터 준비
BUSI 데이터를 받아 아래처럼 배치
```
data/Dataset_BUSI_with_GT/
├── normal/
├── benign/       # benign (1).png, benign (1)_mask.png ...
└── malignant/
```

### 3. 학습
```bash
python train.py --train-dir data/Dataset_BUSI_with_GT/normal
# → outputs/vae.weights.h5, loss_curve.png
```

### 4. 평가
```bash
python evaluate.py --test-dir data/Dataset_BUSI_with_GT/benign --mask-dir data/Dataset_BUSI_with_GT/benign
# → outputs/eval/test_results.png, dice.csv
```

---

## 🔧 주요 트러블슈팅

### Colab 노트북 → 로컬 스크립트 전환
- **문제**: Google Drive 경로, `cv2_imshow` 등 Colab 전용 코드라 로컬에서 실행 불가
- **해결**: 경로를 명령행 인자로 분리, 시각화는 이미지 파일로 저장하도록 변경

### Keras 3 커스텀 모델 메서드 이름 충돌
- **문제**: VAE 클래스에 `_losses` 메서드를 정의하자 `TypeError: 'TrackedList' object is not callable` 발생
- **해결**: Keras 내부 속성 이름과 겹치지 않도록 메서드 이름 변경

### 영상 – GT 마스크 짝 불일치
- **문제**: `os.listdir`는 파일 순서를 보장하지 않아 영상과 마스크가 엇갈려 비교될 수 있음
- **해결**: `benign (1).png` ↔ `benign (1)_mask.png`처럼 파일명 기준으로 매칭, 마스크가 여러 장이면 합쳐서 사용

---

## 📝 한계 및 향후 개선 방향

- 학습 데이터가 normal 133장으로 적음 → 데이터 증강 또는 추가 정상 데이터 필요
- latent 차원 2는 초음파 영상의 다양성을 표현하기에 작음 → latent 차원 확대 실험 필요
- 평균 밝기 기반 임계값은 단순한 규칙 → Otsu, percentile 기반 임계값 등과 비교 필요
- 스페클 노이즈로 인한 잡음 검출 → 이진화 후 morphology 연산 등 후처리 추가 고려

---

## 🙏 참고 / 데이터 출처

- Keras Example — [Variational AutoEncoder](https://keras.io/examples/generative/vae/)
- Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020).
  **Dataset of breast ultrasound images.** *Data in Brief*, 28, 104863.
