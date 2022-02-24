cd mask
python train.py --model Resnet18 --epochs 10 --name exp_adam --optimizer 'Adam'
python inference.py --model Resnet18 --name exp_adam
cd ../gender
python train.py --model Resnet18 --epochs 10 --name exp_adam --optimizer 'Adam'
python inference.py --model Resnet18 --name exp_adam
cd ../age
python train.py --model Resnet18 --epochs 10 --name exp_adam --optimizer 'Adam'
python inference.py --model Resnet18 --name exp_adam
cd
python submission.py --name exp_adam