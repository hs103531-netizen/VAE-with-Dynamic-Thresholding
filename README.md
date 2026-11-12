# VAE with Dynamic Thresholding

정상 유방 초음파 영상만으로 VAE를 학습시키고, 재구성 오차를 이용해 병변 위치를 찾아보는 프로젝트입니다.

병변 라벨 없이도 이상 부위를 잡아낼 수 있을지 궁금해서 시작했습니다.
VAE는 정상 조직만 보고 학습했기 때문에, 병변이 있는 영상을 넣으면 그 부분은 제대로 복원하지 못합니다.
그래서 원본과 복원 영상의 차이를 보면 병변 근처가 두드러지게 됩니다.

## 동작 방식

1. BUSI 데이터셋의 normal 영상을 256×256 흑백으로 맞춰 VAE를 학습합니다.
2. benign / malignant 영상을 넣어 복원하고, 복원 영상에서 원본을 뺀 차이 영상을 만듭니다.
3. 차이 영상을 이진화할 때 고정된 값 대신 원본 영상의 평균 밝기를 임계값으로 씁니다.
   초음파는 장비나 게인 설정에 따라 영상마다 밝기 차이가 커서, 임계값도 영상마다 따로 정하도록 했습니다.
4. 이진화 결과를 GT 마스크와 비교해 Dice 계수로 평가합니다.

모델은 Keras 공식 VAE 예제를 바탕으로 입력 크기를 256×256에 맞게 바꿨습니다.
Conv 레이어 2개로 된 인코더와 2차원 잠재 공간을 쓰고, 손실은 BCE + KL divergence입니다.

## 실행

```bash
pip install -r requirements.txt

python train.py --train-dir data/Dataset_BUSI_with_GT/normal
python evaluate.py --test-dir data/Select --mask-dir data/Select_Mask
```

데이터는 [Kaggle BUSI 데이터셋](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)을 받아 `data/` 아래에 두면 됩니다.
마스크는 파일명으로 짝을 찾기 때문에(`benign (121).png` → `benign (121)_mask.png`) 영상과 같은 폴더에 있어도 상관없습니다.

학습이 끝나면 `outputs/`에 가중치와 loss 그래프가 저장되고, 평가를 돌리면 `outputs/eval/`에 결과 이미지와 영상별 Dice 점수(`dice.csv`)가 저장됩니다.
epoch, 학습률 같은 설정은 `python train.py -h`로 확인할 수 있습니다.

## 파일 구성

- `train.py`, `evaluate.py`: 학습 / 평가 실행
- `vae_anomaly/`: 모델, 데이터 로딩, 임계값 처리 코드
- `Variational_Auto_Encoder_Anomaly.ipynb`: 처음 Colab에서 실험했던 노트북

## 참고

- Keras example, [Variational AutoEncoder](https://keras.io/examples/generative/vae/)
- Al-Dhabyani W. et al., *Dataset of breast ultrasound images*, Data in Brief, 2020.
