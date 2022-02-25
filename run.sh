cd mask
python train.py --model Resnet18 --epochs 15 --name exp2
python inference.py --model Resnet18 --name exp2
cd ../gender
python train.py --model Resnet18 --epochs 15 --name exp2
python inference.py --model Resnet18 --name exp2
cd ../age
python train.py --model Resnet18 --epochs 15 --name exp2
python inference.py --model Resnet18 --name exp2
cd ..
python submission.py --name exp2 --mask_name exp2 --gender_name exp2 --age_name exp2