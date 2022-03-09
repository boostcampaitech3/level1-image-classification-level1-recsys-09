# level1-image-classification-level1-recsys-09

## ❗ 주제 설명

- COVID-19 Pandemic 상황 속 마스크 착용 유무 판단 시스템 구축
- 마스크 착용 여부, 성별, 나이 총 세가지 기준에 따라 총 18개의 class로 구분하는 모델



## 👋 팀원 소개

|                                                  [김혜지](https://github.com/h-y-e-j-i)                                                   |                                                                          [이아현](https://github.com/ahyeon0508)                                                                           |                                                 [김동우](https://github.com/dongwoo338)                                                  |                                                                        [김은선](https://github.com/sun1187)                                                                         |                                                                         [김연요](https://github.com/arkdusdyk)                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![Avatar](https://user-images.githubusercontent.com/69205130/157381112-6343be93-9a26-4778-be7d-cc038f32b459.png)](https://github.com/h-y-e-j-i) | [![Avatar](https://user-images.githubusercontent.com/69205130/157381123-15a8abd6-3dac-4dc1-9aae-d61e94cd1d04.png)](https://github.com/ahyeon0508) | [![Avatar](https://user-images.githubusercontent.com/69205130/157381094-72f2de15-491e-4a4c-9954-701bf924d41b.jpg)](https://github.com/dongwoo338) | [![Avatar](https://user-images.githubusercontent.com/69205130/157381102-fedbcca1-b9e8-47d6-aba4-4ae3ac182a6f.png)](https://github.com/sun1187) | [![Avatar](https://user-images.githubusercontent.com/69205130/157381074-7d91c0e9-756a-4d23-954f-aa43e0688b30.png)](https://github.com/arkdusdyk) |



## 🔨 Installation

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



## ✍ Function Description

`model.py`: EfficientNet-b4와  GoogLeNet을 Ensemble하여 모델링

`dataset.py`: data augmentation, labeling 등 model training에 사용되는 dataset 생성

`loss.py`: cross entropy, f1 score, arcface를 이용해 loss 값을 계산

`train.py`: model을 사용자가 지정한 parameter에 따라 실행하여 training


## 🏢 Structure

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


## ⚙️ Training 명령어

```python
python train.py --model 'Ensemble' --TTA True --name 'final model' --epoch 3
```

![image](https://user-images.githubusercontent.com/44939208/157379480-737623fe-8237-47bc-8c4a-03897a8fd3e9.png)


## 🖼️ 실행 결과

| 모델명 | F1-Score | Accuracy | 최종 순위 |
| --- | --- | --- | --- |
| EfficientNet-b4 + GoogLeNet | 0.7269 | 77.3016 | private 35등 |


## 📜 참고자료

[EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

[GoogLeNet](https://pytorch.org/vision/stable/_modules/torchvision/models/googlenet.html)

