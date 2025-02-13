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

from datasets import ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA
from utils import get_model_from_sd, test_model_on_dataset, ModelWrapper, maybe_dictionarize_batch, get_simclr_pipeline_transform, RandomGaussianDataset
from utils import cosine_lr, test_wbtrainer_on_dataset, test_wbtrainer_assigner_on_dataset, test_edm_on_dataset
from zeroshot import zeroshot_classifier
from imagenet_classnames import openai_classnames
from openai_imagenet_template import openai_imagenet_template
import pdb
import copy
import random
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
    return parser.parse_args()

args = parse_arguments()
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
def create_dog_dataset(dataset, classnames, output_path):
    """
    강아지 클래스에 해당하는 데이터를 필터링하고 저장합니다.
    """
    dog_keywords = ["hound", "terrier", "retriever", "spaniel", "bulldog", "collie", "corgi", "poodle", "shepherd", "mastiff", "pinscher"]
    dog_classes = [cls for cls in classnames if any(keyword in cls.lower() for keyword in dog_keywords)]
    print(f"총 {len(dog_classes)}개의 강아지 관련 클래스가 발견되었습니다.")
    
    class_to_idx = {cls: idx for idx, cls in enumerate(dog_classes)}
    filtered_data = []
    
    for img, label, path in dataset.test_dataset:  # ImageFolderWithPaths는 (img, label, path)를 반환
        if label in class_to_idx.values():
            filtered_data.append((img, label, path))
    
    print(f"총 {len(filtered_data)}개의 강아지 데이터가 필터링되었습니다.")
    
    # 데이터 저장
    torch.save({'data': filtered_data, 'classes': dog_classes}, output_path)
    print(f"강아지 데이터셋이 {output_path}에 저장되었습니다.")

# 강아지 데이터셋 생성 및 저장
imagenet_dataset = ImageNet(preprocess, '/mlainas/bubble3jh/data', 1000, 8)
output_path = "./dataset/dog_imagenet.pth"
create_dog_dataset(imagenet_dataset, openai_classnames, output_path)