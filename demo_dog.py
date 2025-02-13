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
import torch.nn.functional as F

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

    parser.add_argument(
        "--alpha", 
        type=float, 
        default=1.0, 
        help="Orthogonal surgery parameter alpha."
    )
    
    return parser.parse_args()


def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

def compute_and_save_mean_embedding(model, dataset, save_path="./datasets/dogs_emb.pt"):
    """
    Compute and save the mean embedding for DogImageNet with the correct feature dimensions.
    """
    model.eval()
    loader = dataset.test_loader
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing mean embedding"):
            batch = maybe_dictionarize_batch(batch)
            inputs = batch['images'].to(device)
            if args.half_prec:
                inputs = inputs.half()
            
            # Extract features
            _, embeddings = model(inputs, return_features=True)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # Normalize
            all_embeddings.append(embeddings)
    
    # Concatenate embeddings and compute the mean
    all_embeddings = torch.cat(all_embeddings, dim=0)
    mean_embedding = all_embeddings.mean(dim=0)  # Compute mean along the batch dimension
    mean_embedding = mean_embedding / mean_embedding.norm()  # Normalize

    # Save mean embedding
    torch.save(mean_embedding, save_path)
    print(f"Saved mean embedding to {save_path}.")


def adjust_and_evaluate(model, dataset, mean_embedding, alpha=1.0, half_prec=0):
    """
    Adjust embeddings with orthogonal surgery and evaluate in a single function.
    """
    model.eval()
    loader = dataset.test_loader
    device = 'cuda'
    mean_embedding = mean_embedding.unsqueeze(0)  # (1000,) -> (1, 1000) for broadcasting

    top1, top5, correct1, correct5, n = 0., 0., 0., 0., 0.

    with torch.no_grad():
        # 1. Naive projection
        # for batch_idx, batch in enumerate(tqdm(loader, desc="Adjusting and evaluating embeddings")):
        #     batch = maybe_dictionarize_batch(batch)
        #     inputs, labels = batch['images'].to(device), batch['labels'].to(device)
        #     if half_prec:
        #         inputs = inputs.half()

        #     # Generate original embeddings using the model
        #     logits, features = model(inputs, return_features=True)

        #     # Adjust embeddings using orthogonal surgery
        #     projections = (features @ mean_embedding.T) * mean_embedding
        #     orthogonal_component = features - projections
        #     adjusted_features = projections + alpha * orthogonal_component

        # 2. Geodesic projection
        for batch_idx, batch in enumerate(tqdm(loader, desc="Adjusting and evaluating embeddings")):
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].to(device), batch['labels'].to(device)
            if half_prec:
                inputs = inputs.half()

            # Generate original embeddings using the model
            logits, features = model(inputs, return_features=True)

            # [1] 만약 features가 이미 정규화 되어 있지 않다면 normalize
            #     (보통 임베딩이 L2-normalized인 경우도 많으므로 상황에 따라 생략 가능)
            #     shape: features -> (B, D)
            feat_norm = F.normalize(features, p=2, dim=1)

            # mean_embedding도 마찬가지로 정규화 (shape: (D,) 이라고 가정)
            mean_norm = F.normalize(mean_embedding, p=2, dim=0)

            # [2] 두 벡터 사이의 각도(theta) 계산
            #     feat_norm * mean_norm -> (B, D) * (D,) -> (B, D),  sum(dim=1) -> (B,)
            cos_theta = torch.sum(feat_norm * mean_norm, dim=1)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 각도 계산 안전성 확보
            theta = torch.acos(cos_theta)

            # [3] sin(theta), sin((1-alpha)*theta), sin(alpha*theta) 계산
            sin_theta = torch.sin(theta)

            # sin(theta) = 0에 대한 예외 처리 (f와 m이 같은 방향이거나 정반대 방향일 때)
            # 필요 시 epsilon 처리 등
            epsilon = 1e-7
            sin_theta = sin_theta + epsilon

            # slerp = sin((1-alpha)*theta)/sin(theta)*feat_norm + sin(alpha*theta)/sin(theta)*mean_norm
            sin_part_f = torch.sin((1 - alpha) * theta) / sin_theta
            sin_part_m = torch.sin(alpha * theta) / sin_theta

            # shape 일치 위해 unsqueeze(1)로 (B,1) 형태 -> (B,1)*(B,D) broadcasting
            sin_part_f = sin_part_f.unsqueeze(1)
            sin_part_m = sin_part_m.unsqueeze(1)

            adjusted_features = sin_part_f * feat_norm + sin_part_m * mean_norm
            # Pass adjusted embeddings through the classification head
            adjusted_logits = model.classification_head(adjusted_features)

            # Project labels if necessary
            y = labels
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                adjusted_logits = projection_fn(adjusted_logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(labels, device)

            pred_top1 = adjusted_logits.argmax(dim=1, keepdim=True).to(device)
            pred_top5 = adjusted_logits.topk(5, dim=1).indices.to(device)

            # Top-1 accuracy
            correct_current1 = pred_top1.eq(y.view_as(pred_top1)).sum().item()
            correct1 += correct_current1

            # Top-5 accuracy
            correct_current5 = sum(y[i] in pred_top5[i] for i in range(len(y)))
            correct5 += correct_current5

            n += y.size(0)

            if batch_idx % 20 == 0:
                percent_complete = 100.0 * batch_idx / len(loader)
                print(
                    f"[{percent_complete:.0f}% {batch_idx}/{len(loader)}]\t"
                    f"Top-1 Acc: {100 * (correct1/n):.2f}\t"
                    f"Top-5 Acc: {100 * (correct5/n):.2f}"
                )

    top1 = correct1 / n
    top5 = correct5 / n

    return top1, top5



if __name__ == "__main__":
    args = parse_arguments()
    set_seed(0)

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

    # Load model paths
    args.zs_path = args.zs_path or model_paths[args.model]['zs_path']
    args.ft_path = args.ft_path or model_paths[args.model]['ft_path']

    # 1. Load the backbone model & dataset
    if "eva" in args.model:
        base_model = timm.create_model(args.model, pretrained=True, num_classes=1000).to(device)
        data_config = timm.data.resolve_model_data_config(base_model)
        preprocess = timm.data.create_transform(**data_config, is_training=False)
        train_preprocess = timm.data.create_transform(**data_config, is_training=True) if args.data_aug else preprocess
        image_sz = 224
    else:
        base_model, preprocess = clip.load(args.model, 'cpu', jit=False)
        image_sz = base_model.visual.input_resolution
        train_preprocess = preprocess

    base_model.eval()

    # Load dataset
    dog_dataset = DogImageNet(preprocess, args.data_location, args.batch_size, args.workers)
    mean_embedding_path = "./datasets/dogs_emb.pt"

    # Load model
    zs_state_dict = torch.load(args.zs_path, map_location=torch.device('cuda'), weights_only=True)
    model = get_model_from_sd(zs_state_dict, base_model)
    if args.half_prec:
        convert_weights(model)
    model.eval()

    # 2. Compute mean embedding
    if not os.path.exists(mean_embedding_path):
        compute_and_save_mean_embedding(model, dog_dataset, save_path=mean_embedding_path)
    mean_embedding = torch.load(mean_embedding_path, weights_only=True).to(device)

    # 3. Perform orthogonal surgery or standard evaluation
    if args.orthogonal_surgery:
        print("Performing orthogonal surgery...")
        if args.alpha == -1:
            for alpha in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
                top1, top5 = adjust_and_evaluate(
                    model, dog_dataset, mean_embedding, alpha=alpha, half_prec=args.half_prec
                )
                print(f"Alpha: {alpha:.2f} - Top-1 Accuracy: {top1:.3f}%, Top-5 Accuracy: {top5:.3f}%")
        else:
            top1, top5 = adjust_and_evaluate(
                model, dog_dataset, mean_embedding, alpha=args.alpha, half_prec=args.half_prec
            )
            print(f"Orthogonal Surgery Top-1 Accuracy: {top1:.3f}%")
            print(f"Orthogonal Surgery Top-5 Accuracy: {top5:.3f}%")
    else:
        print("Performing standard evaluation...")
        top1, top5 = test_model_on_dataset(model, dog_dataset, rep_mask_depth=0, clip_flag=True, half_prec=args.half_prec)
        print(f"Standard Top-1 Accuracy: {top1:.3f}%")
        print(f"Standard Top-5 Accuracy: {top5:.3f}%")

