python mask/train.py --model Resnet18 --epochs 15 --name exp2
python mask/inference.py --model Resnet18 --name exp2
python gender/train.py --model Resnet18 --epochs 15 --name exp2
python gender/inference.py --model Resnet18 --name exp2
python age/train.py --model Resnet18 --epochs 15 --name exp2
python age/inference.py --model Resnet18 --name exp2
python ../submission.py --name exp2 --mask_name exp2 --gender_name exp2 --age_name exp2