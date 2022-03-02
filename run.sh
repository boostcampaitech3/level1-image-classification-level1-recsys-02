cd age
python train.py --model EfficientNet_b1 --epochs 12 --name crop_b1_adamw_v2 --optimizer AdamW --criterion focal
python inference.py --model EfficientNet_b1 --name crop_b1_adamw_v2
cd ../gender
python train.py --model EfficientNet_b1 --epochs 12 --name crop_b1_adamw_v2 --optimizer AdamW --criterion label_smoothing
python inference.py --model EfficientNet_b1 --name crop_b1_adamw_v2
cd ../mask
python train.py --model EfficientNet_b1 --epochs 10 --name crop_b1_adamw_v2 --optimizer AdamW --criterion label_smoothing
python inference.py --model EfficientNet_b1 --name crop_b1_adamw_v2
cd
python submission.py --name crop_b1_adamw_v2