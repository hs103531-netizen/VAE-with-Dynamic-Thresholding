# VAE with Dynamic Thresholding

BUSI 유방 초음파 데이터로 해본 VAE 기반 이상 탐지.
정상 영상으로만 학습 → 병변 영상 재구성 → 원본이랑 차이 나는 부분을 병변으로 봄.

## Method

- 학습 데이터: normal 영상 (256x256, grayscale)
- 차이 영상: `recon - original`
- 이진화: 영상별 평균 밝기를 threshold로 사용 (dynamic threshold)
  - 초음파는 영상마다 밝기 편차가 커서 고정 threshold 대신 씀
- 평가: GT mask와 Dice coefficient

모델은 Keras VAE 예제 기반, 입력 크기만 256x256으로 수정.

## Usage

```bash
pip install -r requirements.txt

python train.py --train-dir data/Dataset_BUSI_with_GT/normal
python evaluate.py --test-dir data/Select --mask-dir data/Select_Mask
```

- 데이터: [BUSI (Kaggle)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- 결과는 `outputs/`에 저장 (가중치, loss 그래프, 결과 이미지, `dice.csv`)
- 옵션: `python train.py -h`

## Files

```
train.py, evaluate.py
vae_anomaly/        모델, 데이터 로딩, threshold
Variational_Auto_Encoder_Anomaly.ipynb   Colab 원본
```

## Reference

- [Keras VAE example](https://keras.io/examples/generative/vae/)
- Al-Dhabyani et al., Dataset of breast ultrasound images, Data in Brief, 2020
