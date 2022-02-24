# 마스크 착용 상태 분류(나중에 Read.me 잘 정리해서 예쁘게 구성하기!)

## Folder & File description
**code**: 18 Label 분류 모델 폴더<br>
**mask**: 마스크 착용 분류 모델 폴더<br>
**gender**: 성별 분류 모델 폴더<br>
**age**: 연령대 분류 모델 폴더<br>
**submission**: 제출 csv 파일 폴더<br>
<br>
**submission.py**: 모델별 output csv파일을 18개 Label에 맞게 최종 csv파일로 재구성하는 py파일<br>
**run.sh**: 3개 모델을 순차적으로 돌리고 최종 산출 output csv파일을 submission에 저장하는 파일<br>
**remove.sh**: 지우고 싶은 실험을 한 번에 지우게 하는 파일<br>
<br>
## 사용법
<br>
run.sh 파일에서 모든 명령어에서의 --name 인자를 원하는 실험명으로 수정(예시: --name exp)<br>
서버 터미널에서 아래 명령어 사용
```bash
sh run.sh
```
submission폴더에서 실험명.csv 파일 확인<br>
<br>
해당 실험과 관련된 모델 파일, csv파일들을 지우고 싶으면<br>
remove.sh안의 모든 명령어에서 경로 마지막 부분(csv같은 확장자명 제외)을 지우고 싶은 실험명으로 수정(예시: /exp 혹은 /exp.csv)<br>
서버 터미널에서 아래 명령어 사용<br>
'''
sh remove.sh
'''
<br>
<br>
