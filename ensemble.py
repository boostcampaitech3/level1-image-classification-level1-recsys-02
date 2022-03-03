# Hard Voting

import argparse
import pandas as pd
from glob import glob
from pathlib import Path
import re
import os

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mask', type=str, default='mask')
    parser.add_argument('--gender', type=str, default='gender')
    parser.add_argument('--age', type=str, default='age')
    parser.add_argument('--submission', type=str, default=False)
    parser.add_argument('--name', type=str, default='submission')
    args = parser.parse_args()

if not args.submission:
    hardvoting(data_dir=f'output/{args.mask}', save_file_name=args.mask)
    hardvoting(data_dir=f'output/{args.gender}', save_file_name=args.gender)
    hardvoting(data_dir=f'output/{args.age}', save_file_name=args.age)
else:
    hardvoting(f'output/submission', args.name)

print('Ensemble Done!')