import argparse
import os
import wget
import torch
import clip
import json
import operator
import time
from timm.data.transforms_factory import transforms_imagenet_train
import timm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from clip.model import convert_weights

from datasets import ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA, DogImageNet
from utils import get_model_from_sd, test_model_on_dataset, ModelWrapper, maybe_dictionarize_batch, get_simclr_pipeline_transform, RandomGaussianDataset
from utils import cosine_lr, test_wbtrainer_on_dataset, test_wbtrainer_assigner_on_dataset, test_edm_on_dataset
from zeroshot import zeroshot_classifier
from imagenet_classnames import openai_classnames
from openai_imagenet_template import openai_imagenet_template
import pdb
import copy
import random

# from merge_utils import learnable_merging_builder, interpolation, get_block_info
# from merge_optimizer import BlackBoxTrainer, SPSAGC, WhiteBoxTrainer, make_functional, BlackBoxTrainer4Assigner, WhiteBoxTrainer4Assigner
# import nevergrad as ng
# import wandb

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default='/mlainas/bubble3jh/data',
        help="The root directory for the datasets."
    )

    parser.add_argument(
        "--model-location",
        type=str,
        default='./checkpoints',
        help="Where to download the models."
    )

    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1024
    )
    
    parser.add_argument(
        "--half_prec", 
        type=int, 
        default=0
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=8
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default='ViT-B/32', 
        choices=['ViT-B/16', 'ViT-B/32', 'ViT-L/14']
    )
    
    parser.add_argument(
        "--data-aug", 
        type=str, 
        default=''
    )
    
    parser.add_argument(
        "--auto-aug",
        default=None,
        help='Auto aug e.g., rand-m2-n1'
    )
    
    parser.add_argument(
        "--zs_path", 
        type=str, 
        default=''
    )
    
    parser.add_argument(
        "--ft_path", 
        type=str, 
        default=''
    )

    parser.add_argument(
        "--orthogonal-surgery", 
        action='store_true', 
        help="Enable orthogonal surgery for DogImageNet embeddings."
    )
    
    return parser.parse_args()


def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

args = parse_arguments()
set_seed(0) #args.seed

device = 'cuda'

model_paths = {
    'ViT-B/16': {
        'zs_path': './checkpoints/ft_b16_lr_3e-5_minaug_mixup_0.pt',
        'ft_path': './checkpoints/ft_b16_lr_3e-5_minaug_mixup_SEED102.pt'
    },
    'ViT-B/32': {
        'zs_path': './checkpoints/zs_sdy.pt',
        'ft_path': './checkpoints/ft_lr3e-5_minaug_mixup_seed1_sdy.pt'
    },
    'ViT-L/14': {
        'zs_path': './checkpoints/nsml_zs_large.pt',
        'ft_path': './checkpoints/nsml_ft_l14_lr_1e-5_0.1_randaug_mixup_0.0_SEED1.pt'
    }
}

args.zs_path = args.zs_path or model_paths[args.model]['zs_path']
args.ft_path = args.ft_path or model_paths[args.model]['ft_path']

#* 1. load the backbone model & dataset
#TODO other backbone beyond the OpenAI CLIP
if "eva" in args.model:
    base_model = timm.create_model(args.model, pretrained=True, num_classes=1000).to(device)
    data_config = timm.data.resolve_model_data_config(base_model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    if args.data_aug: train_preprocess = timm.data.create_transform(**data_config, is_training=True)
    else:             train_preprocess = preprocess
    image_sz = 224
else:
    base_model, preprocess = clip.load(args.model, 'cpu', jit=False)
    image_sz = base_model.visual.input_resolution 
    if args.data_aug:
        if args.data_aug == 'simclr':
            train_preprocess = get_simclr_pipeline_transform(size=base_model.visual.input_resolution)
        elif args.data_aug == 'timmaug':
            print(f"Use AutoAug from timm: {args.auto_aug}")
            train_preprocess = transforms_imagenet_train(
                    img_size=image_sz,
                    auto_augment=args.auto_aug, # rand-m10-n2
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)
                )
        elif args.data_aug in ['noaug','gnoise']:
            train_preprocess = preprocess
        else:
            raise ValueError(f'{args.data_aug} is not implemented yet')
    else:
        train_preprocess = preprocess

base_model.eval()

#* 2. load the best fine-tuned model and pre-trained model
zs_state_dict = torch.load(args.zs_path, map_location=torch.device('cuda'), weights_only=True)
model = get_model_from_sd(zs_state_dict, base_model)
if args.half_prec:
    convert_weights(model)
model.eval()
    
#* 5. evaluation
eval_batch_size = args.batch_size
eval_batch_size = args.batch_size
eval_datasets = [DogImageNet]  # 평가 데이터셋을 DogImageNet으로 변경
# eval_datasets = [ImageNet]
idx_cnt = None
torch.cuda.empty_cache()
for dataset_cls in eval_datasets:
    print(f'Evaluating on {dataset_cls.__name__}.')
    
    # DogImageNet 데이터셋 생성
    dataset = dataset_cls(preprocess, args.data_location, eval_batch_size, args.workers)
    
    # 모델 평가
    top1, top5 = test_model_on_dataset(model, dataset, rep_mask_depth=0, clip_flag=True, half_prec=args.half_prec)
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")
        
    del dataset; del dataset_cls

    
