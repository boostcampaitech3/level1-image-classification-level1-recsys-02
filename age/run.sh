# cd age # 89.05%
# python train.py --model Resnet18 --epochs 30 --optimizer 'Adam' --lr_decay_step 5 --criterion 'cross_entropy' --name age_adam_crossentropy
# python inference.py --model Resnet18 --name age_adam_crossentropy

# cd age # 89.05%
# python train.py --model Resnet18 --epochs 30 --optimizer 'Adam' --lr_decay_step 5 --criterion 'label_smoothing' --name age_adam_labelsmoothing
# python inference.py --model Resnet18 --name age_adam_labelsmoothing

# cd age # 88.60%
# python train.py --model Resnet18 --epochs 30 --optimizer 'Adam' --lr_decay_step 5 --criterion 'focal' --name age_adam_focal136
# python inference.py --model Resnet18 --name age_adam_focal136

# cd age # 88.57%
# python train.py --model Resnet18 --epochs 30 --optimizer 'Adam' --lr_decay_step 5 --criterion 'focal' --name age_adam_focal118
# python inference.py --model Resnet18 --name age_adam_focal118

# cd age # 88.33%
# python train.py --model Resnet18 --epochs 30 --optimizer 'Adam' --lr_decay_step 5 --criterion 'focal' --name age_adam_focal127
# python inference.py --model Resnet18 --name age_adam_focal127

# cd age # weights = torch.FloatTensor([0.1, 0.1, 0.8]).cuda() 87.17% 
# python train.py --model Resnet18 --epochs 30 --optimizer 'Adam' --criterion 'focal' --name age_adam_focal
# python inference.py --model Resnet18 --name age_adam_focal

# cd age # 86.90%
# python train.py --model Resnet18 --epochs 30 --optimizer 'Adam' --lr_decay_step 5 --criterion 'f1' --name age_adam_f1
# python inference.py --model Resnet18 --name age_adam_f1

# cd age # 84.81%
# python train.py --model Resnet18 --epochs 15 --criterion 'label_smoothing' --name age_sgd_labelsmoothing
# python inference.py --model Resnet18 --name age_sgd_labelsmoothing

# cd age # 82.36%
# python train.py --model Resnet18 --epochs 30 --optimizer 'Adam' --lr_decay_step 5 --criterion 'label_smoothing' --name age_adam_labelsmoothing_augmentation
# python inference.py --model Resnet18 --name age_adam_labelsmoothing_augmentation

# cd age # 80.64%
# python train.py --model Resnet18 --epochs 15 --criterion 'label_smoothing' --name age_sgd_labelsmoothing_augmentation
# python inference.py --model Resnet18 --name age_sgd_labelsmoothing_augmentation

# cd age # weights = torch.FloatTensor([0.3, 0.3, 0.4]).cuda() 79.50%
# python train.py --model Resnet18 --epochs 15 --criterion 'focal' --name age_sgd_focal_augmentation
# python inference.py --model Resnet18 --name age_sgd_focal_augmentation

# cd age # weights = torch.FloatTensor([0.2, 0.3, 0.5]).cuda() 79.32%
# python train.py --model Resnet18 --epochs 15 --criterion 'focal' --name age_sgd_focal235_augmentation
# python inference.py --model Resnet18 --name age_sgd_focal235_augmentation

# cd age # weights = torch.FloatTensor([0.1, 0.1, 0.8]).cuda() 77.11%
# python train.py --model Resnet18 --epochs 15 --criterion 'focal' --name age_sgd_focal118_augmentation
# python inference.py --model Resnet18 --name age_sgd_focal118_augmentation





