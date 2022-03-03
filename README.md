# 마스크 착용 상태 분류
## 카메라로 촬영한 사람 얼굴 이미지의 마스크 착용 여부를 판단하는 Task
### Overview
**COVID-19**의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 과거 높은 사망률을 가진 사스(SARS)나 에볼라(Ebola)와는 달리 COVID-19의 치사율은 오히려 비교적 낮은 편에 속합니다. 그럼에도 불구하고, 이렇게 오랜 기간 동안 우리를 괴롭히고 있는 근본적인 이유는 바로 COVID-19의 강력한 전염력 때문입니다.  
감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 모든 사람들의 **올바른 마스크 착용 상태를 검사**하기 위해서는 추가적인 인적자원이 필요할 것입니다.  
따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.  
### Requirements
```bash
pip install -r requirements.txt
```
### Dataset
마스크를 착용하는 건 COIVD-19의 확산을 방지하는데 중요한 역할을 합니다. 제공되는 이 데이터셋은 사람이 마스크를 착용하였는지 판별하는 모델을 학습할 수 있게 해줍니다. 모든 데이터셋은 아시아인 남녀로 구성되어 있고 나이는 20대부터 70대까지 다양하게 분포하고 있습니다. 간략한 통계는 다음과 같습니다.  
- 전체 사람 명 수 : 4,500  
- 한 사람당 사진의 개수: 7 [마스크 착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장]  
- 이미지 크기: (384, 512)

전체 데이터셋 중에서 60%는 학습 데이터셋으로 활용됩니다.  
**입력값**: 마스크 착용 사진, 미착용 사진, 혹은 이상하게 착용한 사진(코스크, 턱스크)  
<img src="https://github.com/pilkyuchoi/images/blob/main/mask_classification/mask_example.png" width="250" height="300">  
**결과값**: 총 18개의 class를 예측해야합니다. 결과값으로 0~17에 해당되는 숫자가 각 이미지 당 하나씩 나와야합니다.  
**Class Description**: 마스크 착용여부, 성별, 나이를 기준으로 총 18개의 클래스가 있습니다.  
<img src="https://github.com/pilkyuchoi/images/blob/ee0bf9cda119c56b2340a5f04a875313cc9b2a33/mask_classification/class_description.png" width="700" height="600">  
### Pre-processing  
현재 이미지 데이터에는 학습에 불필요한 배경 데이터가 존재합니다. 따라서 얼굴 부분만을 잘라내 사용할 수 있도록 전처리를 진행하였습니다.  
사진에서 사람 얼굴을 Detection하여 Annotation 정보를 추출한 뒤 해당 영역에 맞추어 이미지를 잘라냈습니다.  
Detection에는 RetinaFace 라이브러리를 사용했습니다.(tensorflow==2.2.0 필요, 미처 detect 되지 못한 이미지 약 200장은 사람이 직접 annotate 했습니다)  
preprocessing 폴더 안의 retinaface.ipynb 파일을 실행하면 annotation 정보가 담긴 csv 파일이 생성되고,
해당 파일을 바탕으로 crop.ipynb 파일을 실행하면 기존 데이터에서 얼굴 영역을 잘라낸 파일들을 crop_images 폴더에 저장합니다.
### Model 
<img src="https://github.com/pilkyuchoi/images/blob/main/mask_classification/mask_classification_model.png">

### Train
```bash
python train.py --task [Task명(mask, gender, age 중 택 1)]
```
### Inference
```bash
python inference.py --task [Task명(mask, gender, age 중 택 1)]
```
### Out-Of-Fold Ensemble
```bash
python ensemble.py --mask [k-fold 모델들의 output이 있는 폴더명] --gender [] --age []
```
### Submission
```bash
python submission.py --mask [mask output 파일명] --gender [] --age []
```
*Tip: bash 파일(.sh)을 만들어서 python 명령어 여러 줄을 한 번에 실행할 수 있습니다.  
즉, Mask, Gender, Age 모델을 한 번에 학습시키고 Submission파일을 생성하는 것까지 한 번에 가능합니다.
