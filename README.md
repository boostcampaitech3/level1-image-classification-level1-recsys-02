# 마스크 착용 상태 분류(나중에 Read.me 잘 정리해서 예쁘게 구성하기!)

## Folder & File description
**code**: 18 Label 분류 모델 폴더
**mask**: 마스크(Wear, Incorrect, Not Wear) 착용 분류 모델 폴더<br>
**gender**: 성별(Male, Female) 분류 모델 폴더<br>
**age**: 연령대(<30, >=30 and <60, >=60)분류 모델 폴더<br>
**submission**: Ai stage에 제출할 csv 파일 폴더<br>
<br>
**submission.py**: 모델별 output csv파일을 18개 Label에 맞게 최종 csv파일로 재구성하는 py파일<br>
**run.sh**: 3개 모델을 순차적으로 돌리고 최종 산출 output csv파일을 submission에 저장하는 파일<br>
**remove.sh**: 지우고 싶은 실험을 한 번에 지우게 하는 파일<br>
