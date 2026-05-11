import os

os.system("python train.py --dataset_config configs/dataset/edos.json --parameters_config configs/params/BERT/params_roberta.json")
os.system("python train.py --dataset_config configs/dataset/edos.json --parameters_config configs/params/BERT/params_distillation.json")
os.system("python train.py --dataset_config configs/dataset/edos.json --parameters_config configs/params/BERT/params_albert.json")