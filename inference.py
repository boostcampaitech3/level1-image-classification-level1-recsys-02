import argparse
import os
from importlib import import_module

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset


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
    #OOF Ensemble
    for i in range(n_splits):
        model = load_model(model_dir, num_classes, device, i).to(device)
        model.eval()

        img_root = os.path.join(data_dir, 'crop_images')
        info_path = os.path.join(data_dir, 'info.csv')
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

        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                # TTA
                pred = model(images) / 2
                pred += model(torch.flip(images, dims=(-1,))) / 2 # Horizontal_flip
                pred = pred.argmax(dim=-1)
                preds.extend(pred.cpu().numpy())

        info['ans'] = preds
        info.to_csv(os.path.join(output_dir, f'{args.name}_{i}_fold.csv'), index=False)
    print(f'Inference Done!')


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

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = os.path.join(args.model_dir, args.name)
    output_dir = os.path.join(args.output_dir, args.name)

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
