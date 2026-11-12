# VAE with Dynamic Thresholding

유방 초음파 영상(BUSI)에서 **정상 영상만으로 학습한 VAE(Variational AutoEncoder)** 의 재구성 오차를 이용해
병변(이상) 영역을 비지도 방식으로 찾고, 영상별 **동적 임계값(dynamic threshold)** 으로 이진화하는 프로젝트입니다.
원래 Colab 노트북(`Variational_Auto_Encoder_Anomaly.ipynb`)으로 작성했던 코드를 로컬(VSCode)에서 실행할 수 있는 Python 스크립트로 정리했습니다.

## 방법

1. **학습** – BUSI `normal` 영상(흑백, 256×256)만으로 Convolutional VAE를 학습합니다.
   VAE는 정상 조직의 분포만 학습하므로 병변 부위는 제대로 복원하지 못합니다.
2. **재구성** – 테스트 영상(benign / malignant)을 VAE에 넣어 재구성합니다.
3. **동적 임계값 이진화** – 차이 영상 `D = recon − original` 을 만들고,
   영상마다 **원본의 평균 밝기** `T = mean(original)` 를 임계값으로 사용해 `D > T` 인 픽셀을 이상 영역으로 표시합니다.
   고정 임계값 대신 영상별 밝기에 맞춰 임계값이 바뀌므로, 장비·게인에 따라 밝기가 다른 초음파 영상에 대응할 수 있습니다.
   (초음파에서 병변은 주로 어둡게(저에코) 보이는데, VAE가 이를 주변 정상 조직처럼 밝게 복원하면 `recon − original` 이 커집니다.
   밝은 이상 영역까지 보고 싶으면 `--diff-mode abs` 로 `|recon − original|` 을 쓸 수 있습니다.)
4. **평가** – GT 마스크와 예측 마스크의 Dice coefficient를 계산합니다.

### 모델 구조 (`img_size=256`, `latent_dim=2`)

| | 구성 |
|---|---|
| Encoder | Conv2D(32, s2) → Conv2D(64, s2) → Flatten → Dense(16) → `z_mean`, `z_log_var` (2) → Sampling |
| Decoder | Dense(64·64·64) → Reshape(64,64,64) → Conv2DT(64, s2) → Conv2DT(32, s2) → Conv2DT(1, sigmoid) |
| Loss | Binary cross-entropy (픽셀 합) + KL divergence |
| 학습 | Adam(lr=1e-3), batch 128, 500 epochs |

## 폴더 구조

```
.
├── train.py                 # 정상 영상으로 VAE 학습 → outputs/ 에 가중치·그림 저장
├── evaluate.py              # 재구성 → 동적 이진화 → Dice 계산
├── vae_anomaly/
│   ├── model.py             # Sampling 레이어, Encoder/Decoder, VAE(custom train_step)
│   ├── data.py              # 영상/마스크 로딩 (파일명 기준 정렬·매칭)
│   ├── anomaly.py           # dynamic_threshold, dice_coefficient
│   └── visualize.py         # 결과 그림 저장
├── Variational_Auto_Encoder_Anomaly.ipynb   # 원본 Colab 노트북 (참고용)
├── requirements.txt
└── data/                    # 데이터셋 위치 (git에는 올리지 않음)
```

## 설치

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   /   macOS·Linux: source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow 2.16 이상(Keras 3) 기준으로 작성했습니다 (Python 3.13 + TensorFlow 2.21에서 실행 확인). GPU가 있으면 TensorFlow가 자동으로 사용합니다.

## 데이터 준비

[Breast Ultrasound Images Dataset (BUSI)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)을 받아 `data/` 아래에 둡니다.

```
data/
├── Dataset_BUSI_with_GT/
│   ├── normal/       # 학습용 (normal (1).png, ...)
│   ├── benign/       # benign (1).png, benign (1)_mask.png, ...
│   └── malignant/
├── Select/           # (선택) 평가에 쓸 테스트 영상만 골라 둔 폴더
└── Select_Mask/      # (선택) 위 영상들의 마스크 (benign (121)_mask.png ...)
```

마스크는 **파일명으로 매칭**합니다: `benign (121).png` ↔ `benign (121)_mask.png`.
한 영상에 마스크가 여러 장(`_mask_1.png` 등)이면 합쳐서 사용합니다.
영상과 마스크가 같은 폴더에 있어도 `_mask` 가 들어간 파일은 입력 영상에서 자동으로 제외됩니다.

## 실행

```bash
# 1) 학습
python train.py --train-dir data/Dataset_BUSI_with_GT/normal --epochs 500

# 2) 평가 (노트북에서 쓰던 10장 선택 세트)
python evaluate.py --test-dir data/Select --mask-dir data/Select_Mask --save-masks

# BUSI 원본 폴더 전체로 평가
python evaluate.py --test-dir data/Dataset_BUSI_with_GT/benign --mask-dir data/Dataset_BUSI_with_GT/benign
```

주요 옵션

| 스크립트 | 옵션 | 기본값 | 설명 |
|---|---|---|---|
| train.py | `--img-size` | 256 | 입력 크기 (4의 배수) |
| | `--latent-dim` | 2 | 잠재 공간 차원 |
| | `--epochs` / `--batch-size` / `--lr` | 500 / 128 / 1e-3 | |
| | `--val-split` | 0 | 검증 비율 (>0 이면 val loss도 기록) |
| | `--out-dir` | outputs | 저장 위치 |
| evaluate.py | `--model-dir` | outputs | train.py 결과 폴더 |
| | `--diff-mode` | signed | `signed`: recon−orig, `abs`: \|recon−orig\| |
| | `--save-masks` | off | 예측 마스크 PNG 저장 |

### 출력물

- `outputs/vae.weights.h5`, `config.json`, `history.json`
- `outputs/loss_curve.png`, `outputs/train_reconstruction.png`
- `outputs/eval/test_results.png` – 원본 / 재구성 / 예측 마스크 / GT 마스크
- `outputs/eval/dice.csv` – 영상별 Dice, 터미널에 평균 Dice 출력

VSCode에서는 `train.py` / `evaluate.py` 를 열고 실행(▶)하거나, `.vscode/launch.json` 에 위 인자를 넣어 디버깅할 수 있습니다.

## 참고

- Keras example – [Variational AutoEncoder](https://keras.io/examples/generative/vae/) (fchollet)
- Al-Dhabyani W. et al., *Dataset of breast ultrasound images*, Data in Brief, 2020.
