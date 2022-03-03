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

#3개 조합을 0~17숫자로
def encode_multi_class(mask_label, gender_label, age_label):
    return mask_label * 6 + gender_label * 3 + age_label

#파일 덮어쓰기 방지
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
        dirs = glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mask', type=str, default='mask')
    parser.add_argument('--gender', type=str, default='gender')
    parser.add_argument('--age', type=str, default='age')
    parser.add_argument('--oof', type=str, default=True)
    parser.add_argument('--name', type=str, default='preds')
    args = parser.parse_args()

if args.oof:
    hardvoting(data_dir=f'output/{args.mask}', save_file_name=args.mask)
    hardvoting(data_dir=f'output/{args.gender}', save_file_name=args.gender)
    hardvoting(data_dir=f'output/{args.age}', save_file_name=args.age)
else:
    hardvoting(f'output/preds', args.name)

    # 3개 모델 예측 결과 18 클래스로 변환
    # 파일 불러오기
    mask = pd.read_csv(f'./output/{args.mask}/{args.mask}.csv')
    gender = pd.read_csv(f'./output/{args.gender}/{args.gender}.csv')
    age = pd.read_csv(f'./output/{args.age}/{args.age}.csv')

    #3개 파일 합치기
    df_ = pd.merge(mask,gender, on='ImageID')
    df = pd.merge(df_,age, on='ImageID')

    for i in range(len(df)):
        df.loc[i, 'ans'] = encode_multi_class(df.iloc[i]['ans_x'],df.iloc[i]['ans_y'],df.iloc[i]['ans'])

    #필요한 칼럼만 남기고 저장
    df = df.drop(['ans_x', 'ans_y'], axis=1)
    df.to_csv(os.path.join('./preds', increment_path(args.name)) + '.csv', index=False)


    print('Ensemble Done!')

