import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms 
from torchvision.transforms import *

# import albumentations


# 교차 검증
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import math, random

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

lables_counts = [0]*18
labels_dict = dict()
for i in range(18) : labels_dict[i] = list()

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            transforms.RandomApply(transforms=[CenterCrop((320, 256))], p=0.7),
            Resize(resize, Image.BILINEAR),
            ColorJitter(brightness=0.3, contrast=0.3),
            #torchvision.transforms.RandomAdjustSharpness(sharpness_factor=10, p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=[-1, 20]),
            # transforms.RandomApply(transforms=[AddGaussianNoise()], p=0.7),
            ToTensor(),
            Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)
        ])


    def __call__(self, image):
        return self.transform(image)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 57:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []
    image_labels = []
    
    images_labels_dict = dict()
    images_labels_dict['image_id'] = list()
    images_labels_dict['image_paths'] = list()
    images_labels_dict['mask_labels'] = list()
    images_labels_dict['gender_labels'] = list()
    images_labels_dict['age_labels'] = list()
    images_labels_dict['image_labels'] = list()

    #ages_tmp = [0]*3

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)
                
                # 이미지 추가
                add_image_labels_dict = {
                    # label : mask, 성별(남:0. 여:1), 나이(<30:0, 30>= and <60:1, >=60:2), 현재 label 사진 수)
                    0 : [MaskLabels.MASK, 0, 0, 2745], 
                    1 : [MaskLabels.MASK, 0, 1, 2050], 
                    2 : [MaskLabels.MASK, 0, 2, 415], 
                    3 : [MaskLabels.MASK, 1, 0, 3660], 
                    4 : [MaskLabels.MASK, 1, 1, 4085], 
                    5 : [MaskLabels.MASK, 1, 2, 545], 
                    6 : [MaskLabels.INCORRECT, 0, 0, 549], 
                    7 : [MaskLabels.INCORRECT, 0, 1, 410], 
                    8 : [MaskLabels.INCORRECT, 0, 2, 83], 
                    9: [MaskLabels.INCORRECT, 1, 0, 732], 
                    10 : [MaskLabels.INCORRECT, 1, 1,817], 
                    11 : [MaskLabels.INCORRECT, 1, 2,109], 
                    12 : [MaskLabels.NORMAL, 0, 0,549], 
                    13: [MaskLabels.NORMAL, 0, 1, 410], 
                    14 : [MaskLabels.NORMAL, 0, 2, 83], 
                    15 : [MaskLabels.NORMAL, 1, 0,  732], 
                    16 : [MaskLabels.NORMAL, 1, 1, 817], 
                    17 :[MaskLabels.NORMAL, 1, 2, 109],}
                
                # dict에 label 별로  이미지 정보 저장
                for add_image_labels in add_image_labels_dict.keys():
                    if mask_label==add_image_labels_dict[add_image_labels][0] \
                        and gender_label==add_image_labels_dict[add_image_labels][1] \
                        and age_label==add_image_labels_dict[add_image_labels][2] :

                        labels_dict[add_image_labels].append(img_path)
                        lables_counts[add_image_labels] += 1
                        #self.ages_tmp[age_label] += 1
                        break

        #print(self.ages_tmp)
        MAX_IMAGE = max(lables_counts) # 여러 label 중 가장 많은 이미지의 수 

        lables_counts2 = [0]*18 # label별 이미지 수 확인
        image_id = 0
        for label_idx in range(18): # label 별마다 돈다.
            
            add_image_labels_dict[label_idx][3] = len(labels_dict[label_idx])

            # 최대 이미지 수까지 한 사진당 얼마나 늘릴건지? (균등하게)
            add_num_of_label = math.floor(MAX_IMAGE/add_image_labels_dict[label_idx][3]) #

            # 균등하게 나눈 후 나머지 이미지 수를 무작위로 image id 선택하여 이미지 추가
            add_image_idxs = list()
            for _ in range(MAX_IMAGE%add_image_labels_dict[label_idx][3]):
                while True:
                    rand_idx = random.randint(0, len(labels_dict[label_idx])-1)
                    if rand_idx not in add_image_idxs:
                        add_image_idxs.append(rand_idx)
                        break
            
            for idx in range(len(labels_dict[label_idx])-1):
                # 나머지 이미지를 추가로 늘리는 image id라면 +1
                if idx in add_image_idxs:              
                    lables_counts2[label_idx] += (add_num_of_label+1)
                    for _ in range(add_num_of_label+1):
                        # not StratifiedKFold 용
                        self.image_paths.append(labels_dict[label_idx][idx])
                        self.mask_labels.append(add_image_labels_dict[label_idx][0])
                        self.gender_labels.append(add_image_labels_dict[label_idx][1])
                        self.age_labels.append(add_image_labels_dict[label_idx][2])
                        self.image_labels.append(label_idx)
                        # StratifiedKFold 용
                        self.images_labels_dict['image_id'].append(image_id)
                        self.images_labels_dict['image_paths'].append(labels_dict[label_idx][idx])
                        self.images_labels_dict['mask_labels'].append(add_image_labels_dict[label_idx][0])
                        self.images_labels_dict['gender_labels'].append(add_image_labels_dict[label_idx][1])
                        self.images_labels_dict['age_labels'].append(add_image_labels_dict[label_idx][2])
                        self.images_labels_dict['image_labels'].append(label_idx)
                        
                        image_id += 1

                else:
                    lables_counts2[label_idx] += add_num_of_label
                    for _ in range(add_num_of_label):
                        # not StratifiedKFold 용
                        self.image_paths.append(labels_dict[label_idx][idx])
                        self.mask_labels.append(add_image_labels_dict[label_idx][0])
                        self.gender_labels.append(add_image_labels_dict[label_idx][1])
                        self.age_labels.append(add_image_labels_dict[label_idx][2])
                        self.image_labels.append(label_idx)
                        # StratifiedKFold 용
                        self.images_labels_dict['image_id'].append(image_id)
                        self.images_labels_dict['image_paths'].append(labels_dict[label_idx][idx])
                        self.images_labels_dict['mask_labels'].append(add_image_labels_dict[label_idx][0])
                        self.images_labels_dict['gender_labels'].append(add_image_labels_dict[label_idx][1])
                        self.images_labels_dict['age_labels'].append(add_image_labels_dict[label_idx][2])
                        self.images_labels_dict['image_labels'].append(label_idx)
                        
                        image_id += 1
        # print(lables_counts2) # 데이터 확인
        # pd.DataFrame(self.images_labels_dict).to_csv('/opt/ml/code/baseline/train_add_data.csv', index=False, mode='w')


    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set
    
    def K_fold_Split_dataset(self) :
        train_splits_list, test_splits_list = [], []
        spliter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for train_idx, test_idx in spliter.split(self.images_labels_dict['image_id'], self.images_labels_dict['image_labels']):
            train_splits_list.append(Subset(self, train_idx))
            test_splits_list.append(Subset(self, test_idx))
        return train_splits_list, test_splits_list
    
    def get_trainset_add(self):
        return self.images_labels_dict


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles) # 폴더 갯수
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir) # 폴더 목록
        profiles = [profile for profile in profiles if not profile.startswith(".")] # 005284_male_Asian_22
        split_profiles = self._split_profile(profiles, self.val_ratio) # (0 ~ 폴더갯수) 수에서 랜덤으로 일정 수의 숫자 선택하여 train, val로 분류

        cnt = 0
        # phase : train or val 
        # indices : 랜덤으로 선택한 숫자들
        for phase, indices in split_profiles.items(): 
            for _idx in indices: # 숫자들을 하나씩 가져옵니다
                profile = profiles[_idx] # 해당 숫자 index에 해당하는 폴더를 가져옵니다
                img_folder = os.path.join(self.data_dir, profile) # 폴더의 경로를 가져옵니다
                for file_name in os.listdir(img_folder): # 폴더 안에 있는 사진 파일 이름을 가져옵니다
                    _file_name, ext = os.path.splitext(file_name) 
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg) # 사진 파일의 경로를 가져옵니다
                    mask_label = self._file_names[_file_name] # 파일 이름(mask1, mask2..)에 따라 label을 가져옵니다

                    id, gender, race, age = profile.split("_") # 000004_male_Asian_54, # id_gender_race_age
                    gender_label = GenderLabels.from_str(gender) # gender labeling
                    age_label = AgeLabels.from_number(age) # age labeling

                    # 각각 이미지 경로와 label을 추가합니다
                    self.image_paths.append(img_path) 
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1
        # print("self.indices : ", self.indices)

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()] # train의 cnt, val의 cnt 누적 리스트 


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
