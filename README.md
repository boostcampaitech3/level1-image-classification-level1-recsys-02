### mask, incorrect, normal 별 age, gender 예측
파일 중간에 '##'가 붙은 부분이 파일을 나눌 때 추가한 코드 부분입니다.
### model
모델에서 efficientnet 부분이 제가 시도해본 모델들입니다.
### BCELoss
train 파일과 model 파일에서 BCELoss를 사용하기 위해 바꾼 코드가 있습니다 ('## BCELoss' 주석 부분들입니다)
### 기타
위에 수정한 것들 외에도 crop한 이미지들을 사용해보고 focal loss 사용 등 먼가 시도해 본 것들이 제가 사용한 모델에선 성능이 좋지 않은 경우가 많았는데, 또 다른 경우에는 성능이 좋아질 수도 있다고 생각합니다. 
