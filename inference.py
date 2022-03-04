import argparse
import os
from importlib import import_module
from glob import glob
from pathlib import Path
import re

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
from train import increment_path


def load_model(saved_model, num_classes, device, i):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    model_path = os.path.join(saved_model, f'best_{i+1}_fold.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    task_classes = {
        'mask': 3,
        'gender': 2,
        'age': 3
    }

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = task_classes[args.task]
    oof_pred = None
    n_splits = args.n_splits
    print("Calculating inference results..")
    try:
        info_path = os.path.join(data_dir, 'info.csv')
        info = pd.read_csv(info_path)
        img_root = os.path.join(data_dir, 'crop_images')
        img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    except:
        img_paths = glob(f'{args.data_dir}/*')
        info = pd.DataFrame({'ImageID':img_paths, 'ans': [0 for _ in range(len(img_paths))]})

    #OOF Ensemble
    for i in range(n_splits):
        model = load_model(model_dir, num_classes, device, i).to(device)
        model.eval()

        dataset = TestDataset(img_paths, args.resize)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        preds = []
        # soft Voting
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                # TTA
                pred = model(images) / 2
                pred += model(torch.flip(images, dims=(-1,))) / 2 # Horizontal_flip
                preds.extend(pred.cpu().numpy())
            fold_pred = np.array(preds)

        if oof_pred is None:
            oof_pred = fold_pred / n_splits
        else:
            oof_pred += fold_pred / n_splits
            

    info['ans'] = torch.tensor(oof_pred).argmax(dim=-1)
    info.to_csv(os.path.join(output_dir, f'{args.name}.csv'), index=False)
            
    print(f'Inference Done!')

def hardvoting(data_dir,save_file_name):
    all_files = glob(f'{data_dir}/*.csv') 
    li = []

    # file path에 따라 파일 로드
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        # 결과 저장
        li.append(df['ans'])

    # 결과 종합 concatnate
    frame = pd.concat(li, axis=1, ignore_index=True)
    df['ans'] = frame.mode(axis=1)[0]
    df['ans'] = df['ans'].astype(int)

    df.to_csv(f'{data_dir}/{save_file_name}.csv')

def combine():
        # 3개 모델 예측 결과 18 클래스로 변환
        # 파일 불러오기
        mask = pd.read_csv(f'./output/{args.mask}.csv')
        gender = pd.read_csv(f'./output/{args.gender}.csv')
        age = pd.read_csv(f'./output/{args.age}.csv')

        #3개 파일 합치기
        df_ = pd.merge(mask,gender, on='ImageID')
        df = pd.merge(df_,age, on='ImageID')

        for i in range(len(df)):
            df.loc[i, 'ans'] = MaskBaseDataset.encode_multi_class(df.iloc[i]['ans_x'],df.iloc[i]['ans_y'],df.iloc[i]['ans'])

        #필요한 칼럼만 남기고 저장
        df = df[['ImageID','ans']]
        df.to_csv(os.path.join('./preds', increment_path(args.model)) + '.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=[384, 288], help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='Resnet18', help='model type (default: BaseModel)')
    parser.add_argument('--task', default='mask')
    parser.add_argument('--n_splits', default=5)

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--name', type=str, default='exp')
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    # Combine
    parser.add_argument('--combine', default=False)
    parser.add_argument('--mask', type=str, default='mask')
    parser.add_argument('--gender', type=str, default='gender')
    parser.add_argument('--age', type=str, default='age')

    # Model Ensembel
    parser.add_argument('--model_ensemble', default=False)

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = os.path.join(args.model_dir, args.name)
    output_dir = args.output_dir

    if args.model_ensemble:
        hardvoting(f'preds', 'preds')

    elif args.combine:
        combine()
    else:
        inference(data_dir, model_dir, output_dir, args)