# level1-image-classification-level1-recsys-09

## 1. 주제 설명

- COVID-19 Pandemic 상황 속 마스크 착용 유무 판단 시스템 구축
- 마스크 착용 여부, 성별, 나이 총 세가지 기준에 따라 총 18개의 class로 구분하는 모델


## 👋 팀원 소개

|                                                  [김동우](https://github.com/dongwoo338)                                                   |                                                                          [김연요](https://github.com/arkdusdyk)                                                                           |                                                 [김은선](https://github.com/sun1187)                                                  |                                                                        [김혜지](https://github.com/h-y-e-j-i)                                                                         |                                                                         [이아현](https://github.com/ahyeon0508)                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/73115427?v=4)](https://github.com/dongwoo338) | [![Avatar](https://avatars.githubusercontent.com/u/69205130?s=400&u=a14d779da6a9023a45e60e44072436d356a9461c&v=4)](https://github.com/arkdusdyk) | [![Avatar](https://avatars.githubusercontent.com/u/70509258?v=4)](https://github.com/sun1187) | [![Avatar](https://avatars.githubusercontent.com/u/58590260?v=4)](https://github.com/h-y-e-j-i) | [![Avatar](https://avatars.githubusercontent.com/u/44939208?v=4)](https://github.com/ahyeon0508) |


## 2. Installation

- torch == 1.6.0
- torchvision == 0.7.0
- tensorboard == 2.4.1
- pandas == 1.1.5
- opencv-python == 4.5.1.48
- scikit-learn ~= 0.24.1
- matplotlib == 3.2.1
- efficientnet_pytorch

```python
$ pip install -r $ROOT/level1-image-classification-level1-recsys-09/requirements.txt
```

## 3. Function Description

`model.py`: EfficientNet-b4와  GoogLeNet을 Ensemble하여 모델링

`dataset.py`: data augmentation, labeling 등 model training에 사용되는 dataset 생성

`loss.py`: cross entropy, f1 score, arcface를 이용해 loss 값을 계산

`train.py`: model을 사용자가 지정한 parameter에 따라 실행하여 training

## 4. Structure

```bash
level1-image-classification-level1-recsys-09
│
├── README.md
├── requirements.txt
├── EDA
│   ├── data_EDA.ipynb
│   ├── image_EDA.ipynb
│   └── torchvision_transforms.ipynb
└── python
    ├── dataset.py
    ├── loss.py
    ├── model.py
    └── train.py
```

## 5. Training 명령어

```python
python train.py --model 'Ensemble' --TTA True --name 'final model' --epoch 3
```

![image](https://user-images.githubusercontent.com/44939208/157379480-737623fe-8237-47bc-8c4a-03897a8fd3e9.png)

## 6. 실행 결과

| 모델명 | F1-Score | Accuracy | 최종 순위 |
| --- | --- | --- | --- |
| EfficientNet-b4 + GoogLeNet | 0.7269 | 77.3016 | private 35등 |

## 7. 참고자료

[EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

[GoogLeNet](https://pytorch.org/vision/stable/_modules/torchvision/models/googlenet.html)

