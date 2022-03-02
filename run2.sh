cd mask
python train.py --model EfficientNet_b1 --epochs 10 --name EfficientNet_b1_adamw_2 --optimizer AdamW --criterion label_smoothing
python inference.py --model EfficientNet_b1 --name EfficientNet_b1_adamw_2
cd ../gender
python train0.py --model EfficientNet_b1 --epochs 15 --name EfficientNet_b1_adamw_2 --optimizer AdamW --criterion label_smoothing
python inference0.py --model EfficientNet_b1 --name EfficientNet_b1_adamw_2
python train1.py --model EfficientNet_b1 --epochs 15 --name EfficientNet_b1_adamw_2 --optimizer AdamW --criterion label_smoothing
python inference1.py --model EfficientNet_b1 --name EfficientNet_b1_adamw_2
python train2.py --model EfficientNet_b1 --epochs 15 --name EfficientNet_b1_adamw_2 --optimizer AdamW --criterion label_smoothing
python inference2.py --model EfficientNet_b1 --name EfficientNet_b1_adamw_2
cd ../age
python train0.py --model EfficientNet_b1 --epochs 15 --name EfficientNet_b1_adamw_2 --optimizer AdamW --criterion label_smoothing
python inference0.py --model EfficientNet_b1 --name EfficientNet_b1_adamw_2
python train1.py --model EfficientNet_b1 --epochs 15 --name EfficientNet_b1_adamw_2 --optimizer AdamW --criterion label_smoothing
python inference1.py --model EfficientNet_b1 --name EfficientNet_b1_adamw_2
python train2.py --model EfficientNet_b1 --epochs 15 --name EfficientNet_b1_adamw_2 --optimizer AdamW --criterion label_smoothing
python inference2.py --model EfficientNet_b1 --name EfficientNet_b1_adamw_2
cd
python submission2.py --name EfficientNet_b1_adamw_2