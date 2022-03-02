# train
python train.py --task mask --name mask --criterion focal --epoch 1
python train.py --task gender --name gender --criterion focal --epoch 1
python train.py --task age --name gender --criterion focal_age --epoch 1

# inferencer
python inference.py --task mask --name mask
python inference.py --task gender --name gender
python inference.py --task age --name age

# submission
python submission.py --mask mask --gender gender --age age --name test