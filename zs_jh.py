import numpy as np
import torch
import clip
from tqdm import tqdm
from imagenet_classnames import openai_classnames
from openai_imagenet_template import openai_imagenet_template
import datasets
import torchvision
import argparse
import os
from utils import maybe_dictionarize_batch
from collections import Counter
import json
import wandb

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('/mlainas/bubble3jh/data/'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('./checkpoints'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--custom-template", action="store_true", default=False,
    )
    parser.add_argument(
        "--dataset",  default="ImageNet", 
        help=f"Must be one of {','.join(['ImageNet', 'ImageNetV2', 'ImageNetR', 'ObjectNet', 'ImageNetA'])}"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--model",
        default='ViT-B/32',
        help='Model to use -- e.g. ViT-L/14'
    )
    parser.add_argument(
        "--orthogonal_projection", 
        type=str, 
        default="naive", 
        choices=["naive", "geodesic"],
        help="Projection method (naive or geodesic)."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Alpha parameter for controlling the strength of the projection."
    )
    parser.add_argument(
        "--half_prec",
        type=int,
        default=0,
        help="Use half precision for embeddings if set."
    )
    # [NEW] Which embedding to adjust? {text, image}, default: text
    parser.add_argument(
        "--adjust_target",
        type=str,
        default="text",
        choices=["text", "image"],
        help="Adjust text embeddings or image embeddings? (default: text)"
    )
    parser.add_argument(
        "--target_coef",
        type=float,
        default=1.0,
        help="Target coefficient for combined embedding."
    )
    parser.add_argument(
        "--group_coef",
        type=float,
        default=1.0,
        help="Group coefficient for combined embedding."
    )
    parser.add_argument(
        "--negative_coef",
        type=float,
        default=1.0,
        help="Negative coefficient for combined embedding."
    )
    # [NEW] Add ignore_wandb argument
    parser.add_argument(
        "--ignore_wandb", action="store_true", default=False,
        help="Ignore reporting results to wandb."
    )
    return parser.parse_args()

args = parse_arguments()

print("Torch version:", torch.__version__)

print(clip.available_models())

model, preprocess = clip.load(args.model)

input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

print(f"{len(openai_classnames)} classes, {len(openai_imagenet_template)} templates")

dataset = getattr(datasets, args.dataset)(
        preprocess, 
        location=args.data_location, 
        batch_size=args.batch_size, 
        num_workers=args.workers
    )
dataloader = dataset.test_loader

def zeroshot_classifier(args, model, classnames, templates, device):
    checkpoint_path = f'./checkpoints/{args.dataset}_zs_cls_{args.model.replace("/", "_").replace("-", "_")}.pt'
    
    if os.path.exists(checkpoint_path):
        print('Loading zero-shot classifier.')
        zeroshot_weights = torch.load(checkpoint_path, weights_only=True)
    else:
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template(classname) for template in templates]
                texts = clip.tokenize(texts).cuda() # tokenize
                class_embeddings = model.encode_text(texts) # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        torch.save(zeroshot_weights, checkpoint_path)

    return zeroshot_weights

zeroshot_weights = zeroshot_classifier(args, model, openai_classnames, openai_imagenet_template, 'cuda')

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0, keepdim=True).item() for k in topk]

def collect_top5_predictions(dataloader, model, zeroshot_weights, classnames, device='cuda'):
    top5_predictions = {classname: [] for classname in classnames}
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize_batch(batch)
            images, labels = batch['images'].to(device), batch['labels'].to(device)
            
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            # Get top-5 predictions for each image
            _, top5_indices = logits.topk(5, dim=1, largest=True, sorted=True)
            
            # Convert indices to class names and store in the dictionary
            for label, top5_idx in zip(labels, top5_indices):
                true_classname = classnames[label.item()]
                predicted_classnames = [classnames[idx.item()] for idx in top5_idx]
                top5_predictions[true_classname].extend(predicted_classnames)
    
    # Keep only the top-5 most common predictions for each class
    top5_most_common = {
        classname: [item for item, _ in Counter(predictions).most_common(5)]
        for classname, predictions in top5_predictions.items()
    }
    
    return top5_most_common

# Collect top-5 predictions = Hard Negative Candidates
if not os.path.exists(f'./results/{args.dataset}_top5_predictions_{args.model.replace("/", "_").replace("-", "_")}.json'):
    top5_predictions = collect_top5_predictions(dataloader, model, zeroshot_weights, openai_classnames, 'cuda')
    with open(f'./results/{args.dataset}_top5_predictions_{args.model.replace("/", "_").replace("-", "_")}.json', 'w') as f:
        json.dump(top5_predictions, f, indent=4)
else:
    with open(f'./results/{args.dataset}_top5_predictions_{args.model.replace("/", "_").replace("-", "_")}.json', 'r') as f:
        top5_predictions = json.load(f)

# with torch.no_grad():
#     device = 'cuda'
#     top1, top5, n = 0., 0., 0.
    
#     for i, batch in enumerate(tqdm(dataloader)):
#         batch = maybe_dictionarize_batch(batch)
#         images, labels = batch['images'].to(device), batch['labels'].to(device)
        
#         # predict
#         image_features = model.encode_image(images)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         logits = 100. * image_features @ zeroshot_weights

#         # measure accuracy
#         acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
#         top1 += acc1
#         top5 += acc5
#         n += images.size(0)

# top1 = (top1 / n) * 100
# top5 = (top5 / n) * 100 

# print(f"Top-1 accuracy: {top1:.2f}")
# print(f"Top-5 accuracy: {top5:.2f}")

# 1) group 정보 로드 및 group 매핑(dict) 생성
grouped_imagenet_path = "./results/grouped_imagenet_classes.json"
with open(grouped_imagenet_path, 'r') as f:
    grouped_imagenet_data = json.load(f)

# ImageNet 클래스 -> 그룹명 매핑을 미리 만들어둡니다.
classname_to_group = {}
for group_name, class_list in grouped_imagenet_data.items():
    for cls in class_list:
        # openai_classnames 안에 있는 경우에만 매핑
        if cls in openai_classnames:
            classname_to_group[cls] = group_name

# 2) helper 함수: 주어진 클래스 이름(들)에 대해 text embedding 추출
def get_text_embedding(model, class_name_list, device='cuda'):
    """
    여러 classname에 대한 텍스트 임베딩을 평균낸 결과를 리턴.
    """
    texts = []
    for cn in class_name_list:
        # openai_imagenet_template를 사용해 다양한 prompt 만들기
        templated_texts = [t(cn) for t in openai_imagenet_template]
        texts.extend(templated_texts)
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        embs = model.encode_text(tokens)
    # 개별 임베딩 정규화
    embs = embs / embs.norm(dim=-1, keepdim=True)
    # 평균 후 다시 정규화
    emb_mean = embs.mean(dim=0)
    emb_mean = emb_mean / emb_mean.norm()
    return emb_mean

# 3) 각 클래스마다: target, group, hard negative 임베딩을 구한 뒤 조합
if not os.path.exists(f'./checkpoints/{args.dataset}_embeddings_{args.model.replace("/", "_").replace("-", "_")}.pt'):
    target_embeddings = []
    group_embeddings = []
    negative_embeddings = []

    for class_name in openai_classnames:
        # (a) target embedding
        target_embedding = get_text_embedding(model, [class_name], device='cuda')
        target_embeddings.append(target_embedding)

        # (b) group embedding
        if class_name in classname_to_group:
            group_name = classname_to_group[class_name]
            group_members = [m for m in grouped_imagenet_data[group_name] if m in openai_classnames]
        else:
            group_members = []

        if len(group_members) > 0:
            group_embedding = get_text_embedding(model, group_members, device='cuda')
        else:
            group_embedding = torch.zeros_like(target_embedding)
        group_embeddings.append(group_embedding)

        # (c) hard negatives (top-5)
        if class_name in top5_predictions:
            hard_negatives = top5_predictions[class_name]
            valid_negatives = [cn for cn in hard_negatives if cn in openai_classnames]
            if len(valid_negatives) > 0:
                negative_embedding = get_text_embedding(model, valid_negatives, device='cuda')
            else:
                negative_embedding = torch.zeros_like(target_embedding)
        else:
            negative_embedding = torch.zeros_like(target_embedding)
        negative_embeddings.append(negative_embedding)

    # 텐서로 변환하여 저장
    target_embeddings = torch.stack(target_embeddings)
    group_embeddings = torch.stack(group_embeddings)
    negative_embeddings = torch.stack(negative_embeddings)

    torch.save({
        'target': target_embeddings,
        'group': group_embeddings,
        'negative': negative_embeddings
    }, f'./checkpoints/{args.dataset}_embeddings_{args.model.replace("/", "_").replace("-", "_")}.pt')

embeddings = torch.load(f'./checkpoints/{args.dataset}_embeddings_{args.model.replace("/", "_").replace("-", "_")}.pt', weights_only=True)

# 임베딩 조합 함수
def combine_embeddings(index, embeddings, target_coeff=1.0, group_coeff=1.0, negative_coeff=1.0):
    target_embedding = embeddings['target'][index]
    group_embedding = embeddings['group'][index]
    negative_embedding = embeddings['negative'][index]
    
    combined_emb = (target_coeff * target_embedding +
                    group_coeff * group_embedding -
                    negative_coeff * negative_embedding)
    combined_emb = combined_emb / combined_emb.norm()
    return combined_emb

# 조합된 임베딩 생성
combined_zeroshot_weights = []
for i, class_name in enumerate(openai_classnames):
    combined_emb = combine_embeddings(i, embeddings, target_coeff=args.target_coef, group_coeff=args.group_coef, negative_coeff=args.negative_coef)
    combined_zeroshot_weights.append(combined_emb)

combined_zeroshot_weights = torch.stack(combined_zeroshot_weights, dim=1).cuda()

print(f"[INFO] Combined zero-shot weights shape: {combined_zeroshot_weights.shape}")

# -------------------------------------------------------------------------
# 5) 추가적으로, 새로 만든 combined 임베딩으로 모델 성능 테스트도 해볼 수 있습니다.
#    원한다면 아래와 같이 간단히 test loader에서 Top-1, Top-5 accuracy를 계산할 수 있습니다.
# -------------------------------------------------------------------------
top1_combined, top5_combined, n_combined = 0., 0., 0.
with torch.no_grad():
    for i, batch in enumerate(tqdm(dataloader)):
        batch = maybe_dictionarize_batch(batch)
        images, labels = batch['images'].cuda(), batch['labels'].cuda()
        
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 새로 만든 combined_zeroshot_weights 사용
        logits = 100. * image_features @ combined_zeroshot_weights
        
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        top1_combined += acc1
        top5_combined += acc5
        n_combined += images.size(0)

top1_combined = (top1_combined / n_combined) * 100
top5_combined = (top5_combined / n_combined) * 100

# Initialize wandb if not ignored
if not args.ignore_wandb:
    wandb.init(entity="mlai_medical_ai",project="COE", config=args)
    wandb.config.update(args)
    wandb.run.name = f"{args.model.replace('/', '_').replace('-', '_')}_target_{args.target_coef}_group_{args.group_coef}_negative_{args.negative_coef}"

# Report results to wandb if not ignored
if not args.ignore_wandb:
    wandb.log({"Top-1 accuracy": top1_combined, "Top-5 accuracy": top5_combined})

print(f"[COMBINED] Top-1 accuracy: {top1_combined:.2f}")
print(f"[COMBINED] Top-5 accuracy: {top5_combined:.2f}")