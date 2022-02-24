# 마스크 착용 상태 분류(나중에 Read.me 잘 정리해서 예쁘게 구성하기!)
## 기본 설명
18개 라벨을 한 번에 분류하는 것보다 마스크, 성별, 연령대 분류를 각각 따로 진행하는 것이 성능이 훨씬 좋았습니다.  
따라서 위 3개 태스크의 모델링을 각각 따로 관리하려 합니다.  
Mask, Gender, Age 폴더 안에 각각의 모듈들이 들어있습니다.  
세 폴더 모두 베이스라인 코드인 code 폴더와 동일한 포맷을 기반으로 합니다.

## 폴더 & 파일
### 폴더
**code**: 18 Label 분류 모델 폴더  
**mask**: 마스크 착용 분류 모델 폴더  
**gender**: 성별 분류 모델 폴더  
**age**: 연령대 분류 모델 폴더  
**submission**: 제출 csv 파일 폴더  
### 파일
**submission.py**: 모델별 output csv파일을 18개 Label에 맞게 최종 csv파일로 재구성하는 py파일  
**run.sh**: 3개 모델을 순차적으로 돌리고 최종 산출 output csv파일을 submission에 저장하는 파일  
**remove.sh**: 지우고 싶은 실험을 한 번에 지우게 하는 파일  
  
## 사용법
### 전체 실행
run.sh 파일에서 모든 명령어에서의 --name 인자를 원하는 실험명으로 수정(예시: --name exp)  
서버 터미널에서 아래 명령어 사용
```bash
sh run.sh
```
submission폴더에서 실험명.csv 파일 확인  
### 실험 파일 삭제
해당 실험과 관련된 모델 파일, csv파일들을 지우고 싶으면  
remove.sh안의 모든 명령어에서 경로 마지막 부분(csv같은 확장자명 제외)을 지우고 싶은 실험명으로 수정(예시: /exp 혹은 /exp.csv)  
서버 터미널에서 아래 명령어 사용
```bash
sh remove.sh
```

