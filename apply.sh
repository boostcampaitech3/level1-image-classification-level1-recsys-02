# Efficient_b3
python inference.py --task mask --name mask_e3 --model EfficientNet_b3 --data_dir ${1}
python inference.py --task gender --name gender_e3 --model EfficientNet_b3 --data_dir ${1}
python inference.py --task age --name age_e3 --model EfficientNet_b3 --data_dir ${1}
python inference.py --combine True --mask mask_e3 --gender gender_e3 --age age_e3 --model EfficientNet_b3

# EfficientNet_b1
python inference.py --task mask --name mask_e1 --model EfficientNet_b1 --data_dir ${1}
python inference.py --task gender --name gender_e1 --model EfficientNet_b1 --data_dir ${1}
python inference.py --task age --name age_e1 --model EfficientNet_b1 --data_dir ${1}
python inference.py --combine True --mask mask_e1 --gender gender_e1 --age age_e1 --model EfficientNet_b1

# ResnetCls
python inference.py --task mask --name mask_c --model ResnetCls --data_dir ${1}
python inference.py --task gender --name gender_c --model ResnetCls --data_dir ${1}
python inference.py --task age --name age_c --model ResnetCls --data_dir ${1}
python inference.py --combine True --mask mask_c --gender gender_c --age age_c --model ResnetCls

# model ensemble
python inference.py --model_ensemble True