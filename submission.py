import argparse
import pandas as pd
import glob
from pathlib import Path
import re
import os

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
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mask', type=str, default='mask')
    parser.add_argument('--gender', type=str, default='gender')
    parser.add_argument('--age', type=str, default='age')
    parser.add_argument('--name', type=str, default='submission')
    args = parser.parse_args()

#파일 불러오기
mask = pd.read_csv(f'./output/{args.mask}.csv')
gender = pd.read_csv(f'./output/{args.gender}.csv')
age = pd.read_csv(f'./output/{args.age}.csv')

#3개 파일 합치기
df_ = pd.merge(mask,gender, on='ImageID')
df = pd.merge(df_,age, on='ImageID')

for i in range(len(df)):
    df.loc[i, 'ans'] = encode_multi_class(df.iloc[i]['ans_x'],df.iloc[i]['ans_y'],df.iloc[i]['ans'])

#필요한 칼럼만 남기고 저장
df = df.drop(['ans_x', 'ans_y'], axis=1)
df.to_csv(os.path.join('./submission', increment_path(args.name)) + '.csv', index=False)
print('Submission Done!')