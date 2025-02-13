from collections import Counter
import torch
from tqdm import tqdm
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
import torch.nn.functional as F


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
        help=f"One of ['ImageNet', 'ImageNetV2', 'ImageNetR', 'ObjectNet', 'ImageNetA']"
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
        "--half_prec",
        type=int,
        default=0,
        help="Use half precision for embeddings if set."
    )
    # Which embedding to adjust? {text, image}, default: text
    parser.add_argument(
        "--adjust_target",
        type=str,
        default="text",
        choices=["text", "image"],
        help="Adjust text embeddings or image embeddings? (default: text)"
    )
    # Coefficients for the final text-embedding adjustment
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight to strengthen class-specific features."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight to remove features from negative classes (top-5 neighbors)."
    )
    parser.add_argument(
        "--ignore_wandb", action="store_true", default=False,
        help="Ignore reporting results to wandb."
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of top-k predictions to collect.")
    return parser.parse_args()

args = parse_arguments()

model, preprocess = clip.load(args.model, device="cuda")

print(f"{len(openai_classnames)} classes, {len(openai_imagenet_template)} templates")

dataset = getattr(datasets, args.dataset)(
    preprocess,
    location=args.data_location,
    batch_size=args.batch_size,
    num_workers=args.workers
)
dataloader = dataset.test_loader

def verify_topk_consistency(dataloader, model, zeroshot_weights, classnames, k=5, device='cuda'):
    """
    각 클래스별 top-k 예측이 일관적인지를 확인합니다.
    - 기존 방식과 달리, 클래스 등장 빈도를 고려한 유사도 측정 방식을 사용합니다.
    """
    topk_predictions = {classname: [] for classname in classnames}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Verifying top-k consistency"):
            batch = maybe_dictionarize_batch(batch)
            images, labels = batch['images'].to(device), batch['labels'].to(device)
            
            # Image -> logit
            img_embs = model.encode_image(images)
            img_embs /= img_embs.norm(dim=-1, keepdim=True)
            logits = 100. * img_embs @ zeroshot_weights

            # 상위 k개 클래스 index
            _, topk_idx = logits.topk(k, dim=1)

            # label별로 top-k 클래스 빈도를 누적
            for lbl, idx_list in zip(labels, topk_idx):
                true_cname = classnames[lbl.item()]
                predicted_cnames = [classnames[idx.item()] for idx in idx_list]
                topk_predictions[true_cname].append(predicted_cnames)

    # 클래스별 일관성 평가 (Weighted Jaccard Similarity)
    consistency_results = {}
    for cname, preds_list in topk_predictions.items():
        if len(preds_list) > 1:
            similarity_scores = []
            
            # 모든 top-k 예측을 BoC 벡터로 변환
            boc_vectors = []
            for preds in preds_list:
                boc_vectors.append(Counter(preds))  # 예측 리스트를 빈도 벡터로 변환
            
            # 모든 쌍에 대해 가중 Jaccard 유사도 계산
            for i in range(len(boc_vectors)):
                for j in range(i + 1, len(boc_vectors)):
                    counter1, counter2 = boc_vectors[i], boc_vectors[j]
                    intersection = sum((counter1 & counter2).values())  # 교집합 (최소 등장 횟수 합산)
                    union = sum((counter1 | counter2).values())  # 합집합 (최대 등장 횟수 합산)
                    similarity_scores.append(intersection / union if union > 0 else 1.0)
            
            # 평균 유사도를 저장
            consistency_results[cname] = sum(similarity_scores) / len(similarity_scores)
        else:
            consistency_results[cname] = 1.0  # 샘플이 하나뿐이라면 완벽한 일관성

    return consistency_results


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def check_text_similarity(topk_dict, text_embeddings, classnames, k=5):
    """
    Top-k 클래스들 간 텍스트 임베딩 유사도를 확인합니다.
    """
    similarity_results = {}
    
    for classname, topk_classes in topk_dict.items():
        if len(topk_classes) < k:
            continue  # top-k가 충분하지 않으면 건너뜀
        
        # 해당 클래스와 top-k 클래스들의 임베딩 추출
        embedding_main = text_embeddings[classnames.index(classname)].unsqueeze(0)  # 기준 클래스
        embedding_topk = torch.stack([text_embeddings[classnames.index(cls)] for cls in topk_classes])
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(embedding_main.cpu().numpy(), embedding_topk.cpu().numpy())
        
        # 평균 유사도 저장
        similarity_results[classname] = similarities.mean()
    
    return similarity_results
def zeroshot_classifier(args, model, classnames, templates):
    """
    Returns a simple zero-shot classifier weight matrix (D x #classes)
    using the average embedding across all prompts for each class.
    """
    # e.g. './checkpoints/ImageNet_zs_cls_ViT_B_32.pt'
    checkpoint_path = os.path.join(
        args.model_location,
        f"{args.dataset}_zs_cls_{args.model.replace('/', '_').replace('-', '_')}.pt"
    )
    
    if os.path.exists(checkpoint_path):
        print('Loading zero-shot classifier from:', checkpoint_path)
        zeroshot_weights = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
    else:
        print('Building zero-shot classifier...')
        with torch.no_grad():
            zs_weights = []
            for classname in tqdm(classnames):
                # For each class, tokenize multiple prompts
                texts = [template(classname) for template in templates]
                tokenized = clip.tokenize(texts).cuda()
                class_embs = model.encode_text(tokenized)
                class_embs /= class_embs.norm(dim=-1, keepdim=True)
                # Average across prompts -> single vector per class
                mean_emb = class_embs.mean(dim=0)
                mean_emb /= mean_emb.norm()
                zs_weights.append(mean_emb)
            zeroshot_weights = torch.stack(zs_weights, dim=1).cuda()
        # Save for reuse
        torch.save(zeroshot_weights, checkpoint_path)

    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    """
    Computes top-k accuracy for the given logits and ground-truth labels.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res
zeroshot_weights = zeroshot_classifier(args, model, openai_classnames, openai_imagenet_template)
# for k in [2, 3, 5, 10]:
#     print(f"[Top-{k}]")
#     consistency_results = verify_topk_consistency(dataloader, model, zeroshot_weights, openai_classnames, k=k, device='cuda')
#     # print(consistency_results)
#     print(f'Labrador Retriever: {consistency_results["Labrador Retriever"]}')
#     mean_consistency = np.mean(list(consistency_results.values()))
#     print(mean_consistency)
# exit()
# labrador_index = openai_classnames.index('Labrador Retriever')

# # Labrador Retriever에 대해 실제 모델의 Top-k 예측 확인
# with torch.no_grad():
#     for batch in dataloader:
#         batch = maybe_dictionarize_batch(batch)
#         images, labels = batch['images'].to('cuda'), batch['labels'].to('cuda')

#         # 모델 로짓 계산
#         img_embs = model.encode_image(images)
#         img_embs /= img_embs.norm(dim=-1, keepdim=True)
#         logits = 100. * img_embs @ zeroshot_weights

#         # Top-k 클래스 index 및 이름
#         _, topk_idx = logits.topk(k=2, dim=1)

#         # Labrador Retriever 클래스 샘플만 필터링
#         for img, lbl, topk in zip(images, labels, topk_idx):
#             if lbl.item() == labrador_index:
#                 topk_classes = [openai_classnames[idx.item()] for idx in topk]
#                 print(f"Ground Truth: Labrador Retriever")
#                 print(f"Top-k Predictions: {topk_classes}")
                


# mean_consistency = np.mean(list(consistency_results.values()))
# print("[Top-2] Consistency")
# print(mean_consistency)

from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
from tqdm import tqdm

def compute_target_vs_topk_similarity(dataloader, model, zeroshot_weights, classnames, k=5, device='cuda'):
    """
    각 클래스 라벨에 대해 Top-k에서 target label을 제외한 나머지 클래스들과
    target label 간 텍스트 임베딩 유사도를 계산하고, 평균값을 구함.
    """
    labelwise_similarity = {classname: [] for classname in classnames}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Target-vs-Top-k Similarity"):
            batch = maybe_dictionarize_batch(batch)
            images, labels = batch['images'].to(device), batch['labels'].to(device)

            # Image -> logit
            img_embs = model.encode_image(images)
            img_embs /= img_embs.norm(dim=-1, keepdim=True)
            logits = 100. * img_embs @ zeroshot_weights

            # Top-k index for each image
            _, topk_idx = logits.topk(k, dim=1)

            # 이미지당 target label과 나머지 Top-k의 텍스트 임베딩 유사도 계산
            for lbl, idx_list in zip(labels, topk_idx):
                target_label_idx = lbl.item()
                true_cname = classnames[target_label_idx]
                
                if target_label_idx in idx_list:
                    # target label을 제외한 나머지 Top-k 클래스
                    filtered_idx = [idx.item() for idx in idx_list if idx.item() != target_label_idx]
                else:
                    # target label이 Top-k에 없는 경우, 전체 Top-k 사용
                    filtered_idx = [idx.item() for idx in idx_list]
                
                # 텍스트 임베딩 추출
                target_embedding = zeroshot_weights[:, target_label_idx].cpu().numpy().reshape(1, -1)
                topk_embeddings = zeroshot_weights[:, filtered_idx].t().cpu().numpy()

                # target label과 나머지 Top-k의 코사인 유사도 계산
                similarities = cosine_similarity(target_embedding, topk_embeddings)
                avg_similarity = similarities.mean()  # 평균 유사도 계산

                labelwise_similarity[true_cname].append(avg_similarity)

    # 각 클래스 라벨별 평균 유사도 계산
    labelwise_mean_similarity = {cname: np.mean(similarities) if similarities else 0
                                  for cname, similarities in labelwise_similarity.items()}

    return labelwise_mean_similarity

def compute_target_vs_random_similarity(dataloader, model, zeroshot_weights, classnames, k=5, random_sample_size=5, device='cuda'):
    """
    각 클래스 라벨에 대해 target label과 Top-k를 제외한 나머지 클래스들 중
    랜덤으로 선택한 클래스들의 텍스트 임베딩 간 유사도를 계산하고, 평균값을 구함.
    """
    labelwise_similarity = {classname: [] for classname in classnames}
    num_classes = zeroshot_weights.size(1)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Target-vs-Random Similarity"):
            batch = maybe_dictionarize_batch(batch)
            images, labels = batch['images'].to(device), batch['labels'].to(device)

            # Image -> logit
            img_embs = model.encode_image(images)
            img_embs /= img_embs.norm(dim=-1, keepdim=True)
            logits = 100. * img_embs @ zeroshot_weights

            # Top-k index for each image
            _, topk_idx = logits.topk(k, dim=1)

            # 이미지당 target label과 랜덤 클래스의 텍스트 임베딩 유사도 계산
            for lbl, idx_list in zip(labels, topk_idx):
                target_label_idx = lbl.item()
                true_cname = classnames[target_label_idx]
                
                # Top-k와 target label 제외
                excluded_idx = set(idx_list.cpu().numpy())
                excluded_idx.add(target_label_idx)
                available_idx = list(set(range(num_classes)) - excluded_idx)

                # 랜덤 샘플링
                if len(available_idx) < random_sample_size:
                    random_idx = available_idx  # 샘플링 가능한 수가 부족하면 전체 사용
                else:
                    random_idx = np.random.choice(available_idx, size=random_sample_size, replace=False)

                # 텍스트 임베딩 추출
                target_embedding = zeroshot_weights[:, target_label_idx].cpu().numpy().reshape(1, -1)
                random_embeddings = zeroshot_weights[:, random_idx].t().cpu().numpy()

                # target label과 랜덤 텍스트 임베딩의 코사인 유사도 계산
                similarities = cosine_similarity(target_embedding, random_embeddings)
                avg_similarity = similarities.mean()  # 평균 유사도 계산

                labelwise_similarity[true_cname].append(avg_similarity)

    # 각 클래스 라벨별 평균 유사도 계산
    labelwise_mean_similarity = {cname: np.mean(similarities) if similarities else 0
                                  for cname, similarities in labelwise_similarity.items()}

    return labelwise_mean_similarity

labelwise_mean_topk = compute_target_vs_topk_similarity(dataloader, model, zeroshot_weights, openai_classnames)
labelwise_mean_random = compute_target_vs_random_similarity(dataloader, model, zeroshot_weights, openai_classnames)


classnames_list = list(labelwise_mean_topk.keys())  # or just classnames
topk_values = np.array([labelwise_mean_topk[c] for c in classnames_list])
random_values = np.array([labelwise_mean_random[c] for c in classnames_list])

from scipy.stats import ttest_rel

t_stat, p_value = ttest_rel(topk_values, random_values)

print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")

# for k in [2, 3, 5, 10]:
#     labelwise_mean_similarity = compute_target_vs_topk_similarity(
#         dataloader, model, zeroshot_weights, openai_classnames, k=k, device='cuda'
#     )
#     # print(labelwise_mean_similarity)
#     mean_similarity = np.mean(list(labelwise_mean_similarity.values()))
#     print(f"[Top-{k}]")
#     print(mean_similarity)

def compute_image_vs_topk_similarity(dataloader, model, zeroshot_weights, classnames, k=5, device='cuda'):
    """
    각 이미지의 임베딩과 Top-k 텍스트 임베딩 간 코사인 유사도를 계산하고,
    클래스 라벨별로 평균값을 구함.
    """
    labelwise_similarity = {classname: [] for classname in classnames}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Image-vs-Top-k Similarity"):
            batch = maybe_dictionarize_batch(batch)
            images, labels = batch['images'].to(device), batch['labels'].to(device)

            # Image -> logit
            img_embs = model.encode_image(images)
            img_embs /= img_embs.norm(dim=-1, keepdim=True)  # Normalize image embeddings

            # Top-k index for each image
            logits = 100. * img_embs @ zeroshot_weights
            _, topk_idx = logits.topk(k, dim=1)

            # 이미지당 Image-to-Top-k 텍스트 임베딩의 코사인 유사도 계산
            for img_emb, lbl, idx_list in zip(img_embs, labels, topk_idx):
                true_cname = classnames[lbl.item()]
                # Top-k 텍스트 임베딩 추출
                topk_embeddings = zeroshot_weights[:, idx_list].t().cpu().numpy()
                img_embedding = img_emb.cpu().numpy().reshape(1, -1)  # Target image embedding

                # 코사인 유사도 계산
                similarities = cosine_similarity(img_embedding, topk_embeddings)
                avg_similarity = similarities.mean()  # Top-k와의 평균 유사도

                labelwise_similarity[true_cname].append(avg_similarity)

    # 각 클래스 라벨별 평균 유사도 계산
    labelwise_mean_similarity = {cname: np.mean(similarities) if similarities else 0
                                  for cname, similarities in labelwise_similarity.items()}

    return labelwise_mean_similarity

result = compute_image_vs_topk_similarity(
    dataloader, model, zeroshot_weights, openai_classnames, k=1, device='cuda'
)
print(result)
mean_similarity = np.mean(list(result.values()))
print("[Image-vs-Top-1]")
print(mean_similarity)
