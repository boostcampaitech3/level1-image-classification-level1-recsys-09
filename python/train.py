import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch, torch.nn
import torchvision
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet

from dataset import MaskBaseDataset, TestDataset
from dataset import MaskSplitByProfileDataset
from loss import create_criterion
from torchensemble import VotingClassifier

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from model import MyEnsemble, MyEnsemble2, AlexNet



import math, time

matplotlib.use('Agg')

pretraind_model_name = ['GoogleNet', "ShuffleNet", "DenseNet", "SqueezeNet", 'EfficientNet', "ResNet18", "Resnext50", \
                    "Ensemble", "Ensemble2", "Ensemble3", "Ensemble4", "Ensemble5", "Ensemble6"]

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers):
    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(dataset,
                                        indices=train_idx)
    # 인자로 전달받은 dataset에서 valid_idx에 해당하는 Subset 추출
    val_set   = torch.utils.data.Subset(dataset,
                                        indices=valid_idx)
    
    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True
    )
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False
    )
    
    # 생성한 DataLoader 반환
    return train_loader, val_loader 

def build_model():
    num_classes = 18
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- model  
    if args.model == "Ensemble":
        print(args.model)
        googlenet =  torchvision.models.googlenet(pretrained=True)
        googlenet.fc = torch.nn.Sequential(            
            torch.nn.Linear(in_features=googlenet.fc.in_features, out_features=num_classes),
            torch.nn.LogSoftmax(dim=1)
        )
        efficientnet = EfficientNet.from_pretrained('efficientnet-b4')
        efficientnet._fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=efficientnet._fc.in_features, out_features=num_classes),
            torch.nn.LogSoftmax(dim=1)
        )
    else : 
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes,
        ).to(device)
        model = torch.nn.DataParallel(model)
    return model


def train(data_dir, model_dir, args):
    start_time = time.time()

    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader   
    if args.TTA == True:
        test_img_root = '/opt/ml/input/data/eval'   
        # public, private 테스트셋이 존재하니 각각의 예측결과를 저장합니다.

        # meta 데이터와 이미지 경로를 불러옵니다.
        submission = pd.read_csv(os.path.join(test_img_root, 'info.csv'))
        image_dir = os.path.join(test_img_root, 'images')

        # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
        image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
        test_dataset = TestDataset(image_paths, resize=(128, 96))

        test_loader = DataLoader(
            test_dataset,
            shuffle=False
        )
    #if  args.KFold == False: 
    train_set, val_set = dataset.split_dataset()
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )
    # else : 
    #     train_set, val_set = dataset.K_fold_Split_dataset()

    # -- model
    model = build_model()
    
    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    scheduler =  CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    if  args.KFold == False and args.TTA == False:
        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)
                print()
    elif args.TTA == True:
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits)

        counter = 0
        patience = 10
        accumulation_steps = 2
        best_val_acc = 0
        best_val_loss = np.inf
        oof_pred = None

        labels = [dataset.encode_multi_class(mask, gender, age) for mask, gender, age in zip(dataset.mask_labels, dataset.gender_labels, dataset.age_labels)]

        # K-Fold Cross Validation과 동일하게 Train, Valid Index를 생성합니다. 
        for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, labels)):
            train_loader, val_loader = getDataloader(dataset, train_idx, valid_idx, args.batch_size, num_workers=multiprocessing.cpu_count()//2)

            # -- model
            model = build_model()

            # -- loss & metric
            criterion = create_criterion(args.criterion)
            opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
            optimizer = opt_module(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=5e-4
            )
            # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
            scheduler =  CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

            # -- logging
            logger = SummaryWriter(log_dir=save_dir)
            for epoch in range(args.epochs):
                # train loop
                model.train()
                loss_value = 0
                matches = 0
                for idx, train_batch in enumerate(train_loader):
                    inputs, labels = train_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, labels)

                    loss.backward()
                    
                    # -- Gradient Accumulation
                    if (idx+1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    loss_value += loss.item()
                    matches += (preds == labels).sum().item()
                    if (idx + 1) % args.log_interval == 0:
                        train_loss = loss_value / args.log_interval
                        train_acc = matches / args.batch_size / args.log_interval
                        current_lr = scheduler.get_last_lr()
                        print(
                            f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                            f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                        )
                        logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                        logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                        loss_value = 0
                        matches = 0

                scheduler.step()

                # val loop
                with torch.no_grad():
                    print("Calculating validation results...")
                    model.eval()
                    val_loss_items = []
                    val_acc_items = []
                    figure = None
                    for val_batch in val_loader:
                        inputs, labels = val_batch
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outs = model(inputs)
                        preds = torch.argmax(outs, dim=-1)

                        loss_item = criterion(outs, labels).item()
                        acc_item = (labels == preds).sum().item()
                        val_loss_items.append(loss_item)
                        val_acc_items.append(acc_item)

                        if figure is None:
                            inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                            inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                            figure = grid_image(
                                inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )

                    val_loss = np.sum(val_loss_items) / len(val_loader)
                    val_acc = np.sum(val_acc_items) / len(valid_idx)

                    # Callback1: validation accuracy가 향상될수록 모델을 저장합니다.
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    if val_acc > best_val_acc:
                        print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                        torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                        best_val_acc = val_acc
                        counter = 0
                    else:
                        counter += 1
                    torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                    # Callback2: patience 횟수 동안 성능 향상이 없을 경우 학습을 종료시킵니다.
                    if counter > patience:
                        print("Early Stopping...")
                        break

                    print(
                        f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                        f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                    )

                    logger.add_scalar("Val/loss", val_loss, epoch)
                    logger.add_scalar("Val/accuracy", val_acc, epoch)
                    logger.add_figure("results", figure, epoch)
                    print()
                    
            # 각 fold에서 생성된 모델을 사용해 Test 데이터를 예측합니다. 
            all_predictions = []
            with torch.no_grad():
                for images in test_loader:
                    images = images.to(device)

                    # Test Time Augmentation
                    pred = model(images) / 2 # 원본 이미지를 예측하고
                    pred += model(torch.flip(images, dims=(-1,))) / 2 # horizontal_flip으로 뒤집어 예측합니다. 
                    all_predictions.extend(pred.cpu().numpy())

                fold_pred = np.array(all_predictions)

            # 확률 값으로 앙상블을 진행하기 때문에 'k'개로 나누어줍니다.
            if oof_pred is None:
                oof_pred = fold_pred / n_splits
            else:
                oof_pred += fold_pred / n_splits
    elif args.KFold == True:
        # 5-fold Stratified KFold 5개의 fold를 형성하고 5번 Cross Validation을 진행합니다.
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits)

        counter = 0
        patience = 10
        accumulation_steps = 2
        best_val_acc = 0
        best_val_loss = np.inf

        labels = [dataset.encode_multi_class(mask, gender, age) for mask, gender, age in zip(dataset.mask_labels, dataset.gender_labels, dataset.age_labels)]

        # Stratified KFold를 사용해 Train, Valid fold의 Index를 생성합니다.
        # labels 변수에 담긴 클래스를 기준으로 Stratify를 진행합니다. 
        for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, labels)):
            
            # 생성한 Train, Valid Index를 getDataloader 함수에 전달해 train/valid DataLoader를 생성합니다.
            # 생성한 train, valid DataLoader로 이전과 같이 모델 학습을 진행합니다. 
            train_loader, val_loader = getDataloader(dataset, train_idx, valid_idx, args.batch_size, num_workers=multiprocessing.cpu_count()//2)

            # -- model
            model = build_model()

            # -- loss & metric
            criterion = create_criterion(args.criterion)
            # train_params = [{'params': getattr(model, 'features').parameters(), 'lr': args.lr / 10, 'weight_decay':5e-4},
            #                 {'params': getattr(model, 'classifier').parameters(), 'lr': args.lr, 'weight_decay':5e-4}]
            opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
            optimizer = opt_module(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=5e-4
            )
            # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
            scheduler =  CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

            # -- logging
            logger = SummaryWriter(log_dir=save_dir)
            for epoch in range(args.epochs):
                # train loop
                model.train()
                loss_value = 0
                matches = 0
                for idx, train_batch in enumerate(train_loader):
                    inputs, labels = train_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, labels)

                    loss.backward()
                    
                    # -- Gradient Accumulation
                    if (idx+1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    loss_value += loss.item()
                    matches += (preds == labels).sum().item()
                    if (idx + 1) % args.log_interval == 0:
                        train_loss = loss_value / args.log_interval
                        train_acc = matches / args.batch_size / args.log_interval
                        current_lr = scheduler.get_last_lr()
                        print(
                            f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                            f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                        )
                        logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                        logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                        loss_value = 0
                        matches = 0

                scheduler.step()

                # val loop
                with torch.no_grad():
                    print("Calculating validation results...")
                    model.eval()
                    val_loss_items = []
                    val_acc_items = []
                    figure = None
                    for val_batch in val_loader:
                        inputs, labels = val_batch
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outs = model(inputs)
                        preds = torch.argmax(outs, dim=-1)

                        loss_item = criterion(outs, labels).item()
                        acc_item = (labels == preds).sum().item()
                        val_loss_items.append(loss_item)
                        val_acc_items.append(acc_item)

                        if figure is None:
                            inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                            inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                            figure = grid_image(
                                inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )

                        

                    val_loss = np.sum(val_loss_items) / len(val_loader)
                    val_acc = np.sum(val_acc_items) / len(valid_idx)

                    # Callback1: validation accuracy가 향상될수록 모델을 저장합니다.
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    if val_acc > best_val_acc:
                        print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                        torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                        best_val_acc = val_acc
                        counter = 0
                    else:
                        counter += 1
                    torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                    # Callback2: patience 횟수 동안 성능 향상이 없을 경우 학습을 종료시킵니다.
                    if counter > patience:
                        print("Early Stopping...")
                        break


                    print(
                        f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                        f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                    )
                    logger.add_scalar("Val/loss", val_loss, epoch)
                    logger.add_scalar("Val/accuracy", val_acc, epoch)
                    logger.add_figure("results", figure, epoch)
                    print()
    
    if args.TTA == True:
        output_dir = './output'

        submission['ans'] = np.argmax(oof_pred, axis=1)
        submission.to_csv(os.path.join(output_dir, f'{args.name}.csv'), index=False)

        print('test inference is done!')
    elif args.model in pretraind_model_name:
        ## inference.py 코드 불러오기
        test_dir = '/opt/ml/input/data/eval'
        model_dir = './model'
        output_dir = './output'


        num_classes = MaskBaseDataset.num_classes  # 18
        model = model.to(device)
        model.eval()

        ## data_dir -> test_dir로 수정
        img_root = os.path.join(test_dir, 'images')
        info_path = os.path.join(test_dir, 'info.csv')
        info = pd.read_csv(info_path)

        img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
        dataset = TestDataset(img_paths, args.resize)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        print("Calculating inference results..")
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                pred = model(images)
                pred = pred.argmax(dim=-1)
                preds.extend(pred.cpu().numpy())

        info['ans'] = preds
        info.to_csv(os.path.join(output_dir, f'{args.name}.csv'), index=False)
        print(f'Inference Done!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: CustomAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='arcface', help='criterion type (default: arcface)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--KFold', default=False, type=bool, help='train using StratifiedKFold (default : False)')
    parser.add_argument('--TTA', default=False, type=bool, help='train using TTA (default : False)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)