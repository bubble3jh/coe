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
from utils import maybe_dictionarize_batch, log_map_batch, exp_map_batch
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
        "--method",
        type=str,
        default='simple',
        help="Weight to enhance features."
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

def zeroshot_classifier(args, model, classnames, templates):
    """
    Returns a simple zero-shot classifier weight matrix (D x #classes)
    using the average embedding across all prompts for each class.
    """
    # e.g. './checkpoints/ImageNet_ViT_B_32_text_embs.pt'
    checkpoint_path = os.path.join(
        args.model_location,
        f"{args.dataset}_{args.model.replace('/', '_').replace('-', '_')}_text_embs.pt"
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

def collect_topk_predictions(dataloader, model, zeroshot_weights, classnames, k=5, device='cuda'):
    """
    각 ground-truth label(클래스)마다, 모델이 예측한 top-k 클래스들을 누적한 뒤,
    그 중 가장 많이 등장한 상위 k개를 모아 dictionary 형태로 반환.
    """
    print("[INFO] You are using old function for top-k predictions ...")
    topk_predictions = {classname: [] for classname in classnames}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting top-k predictions"):
            batch = maybe_dictionarize_batch(batch)
            images, labels = batch['images'].to(device), batch['labels'].to(device)
            
            # Image -> logit
            img_embs = model.encode_image(images)
            img_embs /= img_embs.norm(dim=-1, keepdim=True)
            logits = 100. * img_embs @ zeroshot_weights

            # 상위 k개 클래스 index
            _, topk_idx = logits.topk(k, dim=1)

            # label별로 top-k 클래스 이름을 누적
            for lbl, idx_list in zip(labels, topk_idx):
                true_cname = classnames[lbl.item()]
                predicted_cnames = [classnames[idx.item()] for idx in idx_list]
                topk_predictions[true_cname].extend(predicted_cnames)

    # 각 클래스별로 최빈도 상위 k개만 추출
    topk_most_common = {}
    for cname, preds in topk_predictions.items():
        common_list = Counter(preds).most_common(k)
        topk_most_common[cname] = [x[0] for x in common_list]

    return topk_most_common

def collect_topk_indices(dataloader, model, zeroshot_weights, device='cuda', k=5):
    """
    Return:
        - topk_indices: LongTensor of shape (N, k) indicating the top-k class indices per image.
    We do NOT reference ground-truth labels, only the model's predictions.
    """
    model.eval()
    all_topk = []
    with torch.no_grad():
        idx_counter = 0
        for batch in tqdm(dataloader, desc="First pass: collecting top-k"):
            images = batch['images'].to(device)
            # images: (B, C, H, W)

            img_embs = model.encode_image(images)
            img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)  # (B, D)
            
            # zsw: shape (D, #classes) 
            logits = 100. * (img_embs @ zeroshot_weights)  # (B, #classes)
            _, topk_idx = logits.topk(k, dim=1)            # (B, k)
            
            all_topk.append(topk_idx.cpu())  # store on CPU
            idx_counter += images.size(0)

    topk_indices = torch.cat(all_topk, dim=0)  # shape (N, k)
    return topk_indices


# ----------------------------------------------------------------
# 2) Helper to get average text embedding for one or more classnames
# ----------------------------------------------------------------
def get_text_embedding(model, cls_list, device='cuda'):
    """
    Returns the average text embedding across all prompts
    for all classes in `cls_list`.
    """
    texts = []
    for cname in cls_list:
        # multiple prompts
        prompts = [t(cname) for t in openai_imagenet_template]
        texts.extend(prompts)

    tokenized = clip.tokenize(texts).to(device)
    with torch.no_grad():
        embs = model.encode_text(tokenized)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    # average
    mean_emb = embs.mean(dim=0)
    mean_emb = mean_emb / mean_emb.norm()
    return mean_emb

def get_text_embedding_by_idx(model, class_idx, classnames, device='cuda'):
    """
    Convert a class index -> class name -> text embedding.
    If your code uses multiple prompts, average them, etc.
    """
    cname = classnames[class_idx]
    # or your existing 'get_text_embedding(model, [cname])' method
    return get_text_embedding(model, [cname], device=device)  # shape (D,)

# ----------------------------------------------------------------
# 3) Create new "modified" text embeddings using top-5 info
# ----------------------------------------------------------------
# def push_away(e_j, e_others, alpha=1.0, beta=1.0, method='simple'):
#     """
#     Example:
#       e_j' = alpha * e_j - beta * proj_{avg(others)}(e_j)
#     Then normalized. You can refine to your needs.
#     """
#     if method == 'simple':
#         if e_others.ndim == 1:
#             e_others = e_others.unsqueeze(0)  # Ensure (N, D) shape
#         N, D = e_others.shape
#         # Compute means
#         mean_negatives = e_others.mean(dim=0)  # Mean feature of negatives
#         mean_common = (e_j + mean_negatives * N) / (N + 1)  # Approx common feature

#         # Apply direct modification
#         e_j_new = e_j + alpha * mean_common - beta * mean_negatives

#         # Normalize the new embedding
#         e_j_new = e_j_new / e_j_new.norm(dim=0, keepdim=True)
#     elif method == 'qr_proj':
#         """
#         e_j: (D,) 하나의 텍스트 임베딩
#         e_others: (N, D) shape, negative 클래스를 대표하는 임베딩들
#         alpha, beta 각각 e_j 자체 강화, negative 부분 제거 정도 조절
#         """

#         # 차원 맞추기
#         if e_others.ndim == 1:
#             e_others = e_others.unsqueeze(0)  # (1, D)
#         e_j = e_j.squeeze()  # (D,)

#         # ---------------------------
#         # 1) e_others 부분공간의 정규직교 기저 구하기
#         # ---------------------------
#         # e_others가 span하는 부분공간 S를 구하기 위해 QR 분해(Orthonormal basis)
#         # M: (N, D)
#         M = e_others  # shape (N, D)
#         # 필요하면 평균 제거나 정규화 등을 해줄 수도 있음 (문제에 따라)
#         # 여기서는 바로 QR
#         Q, R = torch.linalg.qr(M.T, mode='reduced')  # M.T: (D, N)
#         # Q: (D, k)  <- 부분공간 S를 spanned by columns of Q
#         # R: (k, N)

#         # ---------------------------
#         # 2) e_j를 두 부분으로 분해: 공통성분 p, 고유성분 r
#         # ---------------------------
#         # p = e_j의 S 위로의 투영(projection onto span(Q))
#         #   p = Q * (Q^T e_j)
#         # r = e_j - p
#         #   = S와 수직인 e_j의 부분
#         p = Q @ (Q.T @ e_j)     # S 위 투영
#         r = e_j - p             # S와 수직인 성분

#         # ---------------------------
#         # 3) '진짜 negative' 부분 제거 or 감소
#         # ---------------------------
#         # 만약 "S 중에서도 e_j와 겹치는 성분은 살리되, orthogonal한 성분만 제거"
#         # 하고 싶으면, S에서 e_j와 orthogonal한 방향만 Q에서 추출해야 함.
#         # 하지만 일단 간단히 전체 투영 p는 "공통"으로 보고 살린다고 가정.
#         # 그리고 r(고유 성분)은 alpha로 강화
#         # p(공통 성분)는 gamma로 조절
#         # -> 기존에 "beta * projection_neg" 식으로 직접 빼는 대신
#         #    여기서는 S 내 성분 자체를 얼마나 살릴지/제거할지 gamma로 결정
#         #    (또는 Q에서 e_j와 수직한 기저만 골라서 빼주도록 구현 가능)

#         e_j_new = alpha * r + gamma * p

#         # ---------------------------
#         # 4) 정규화
#         # ---------------------------
#         e_j_new = e_j_new / e_j_new.norm(dim=0, keepdim=True)

#     return e_j_new

# def build_modified_text_embs(
#     model,
#     topk_indices,
#     classnames,
#     alpha=1.0,
#     beta=1.0,
#     method='simple',
#     device='cuda'
# ):
#     """
#     - topk_indices: shape (N, k), e.g. topk_indices[i] -> (k,) class indices
#     - Return a single tensor (N, k, D): the new text embedding for each image's top-k classes.
#     """
#     model.eval()
#     N, k = topk_indices.size()
#     # We'll find dimension D from a single embedding
#     tmp_emb = get_text_embedding_by_idx(model, topk_indices[0, 0].item(), classnames, device=device)
#     D = tmp_emb.shape[0]

#     modified_embs = torch.zeros((N, k, D), dtype=torch.float32)  # store on CPU or GPU?

#     with torch.no_grad():
#         for i in tqdm(range(N), desc="Building modified text embeddings"):
#             class_idx_for_img = topk_indices[i]  # (k,)
            
#             # gather original text embeddings for these k classes
#             orig_embs = []
#             for idx_class in class_idx_for_img:
#                 emb = get_text_embedding_by_idx(model, idx_class.item(), classnames, device=device)
#                 orig_embs.append(emb)
#             orig_embs = torch.stack(orig_embs, dim=0)  # shape (k, D)

#             # for each e_j, push away from the others
#             new_embs = []
#             for j in range(k):
#                 e_j = orig_embs[j]
#                 e_others = torch.cat([orig_embs[:j], orig_embs[j+1:]], dim=0)  # (k-1, D)
#                 e_j_new = push_away(e_j, e_others, alpha=alpha, beta=beta, method=method)
#                 new_embs.append(e_j_new)
            
#             new_embs = torch.stack(new_embs, dim=0)  # (k, D)
#             modified_embs[i] = new_embs.cpu()  # store in CPU, or .to('cuda') if you have memory

#     return modified_embs  # shape (N, k, D)

def build_modified_text_embs(
    class_embeddings,  # (C, D)
    topk_indices,      # (N, k)
    alpha=1.0,
    beta=1.0,
    method='simple',
    device='cuda',
    img_embs=None,
    batch_size=2048   # 원하는 배치 크기를 지정
):
    """
    Vectorized version of building modified text embeddings (batched version)
    """
    N, k = topk_indices.shape
    C, D = class_embeddings.shape

    modified_embs_list = []  # 각 배치 결과를 저장
    cos_sim_list = []        # 각 배치별 cosine similarity를 저장 (평균값으로 나중에 합산)

    # 전체 N개를 배치 단위로 순회합니다.
    for start in tqdm(range(0, N, batch_size), desc="Processing batches"):
        end = min(N, start + batch_size)
        B = end - start  # 현재 배치 크기

        # 현재 배치에 해당하는 topk_indices (shape: (B, k))
        topk_indices_batch = topk_indices[start:end]
        # 해당 배치의 top-k 텍스트 임베딩 추출 (shape: (B, k, D))
        topk_embs = class_embeddings[topk_indices_batch.view(-1)].view(B, k, D).to(device)

        if method == 'simple':
            # 각 그룹별 평균 임베딩 계산 (B, 1, D)
            sum_embs = torch.sum(topk_embs, dim=1, keepdim=True)
            # 각 벡터를 제외한 나머지의 합 (B, k, D)
            sum_others = sum_embs - topk_embs
            mean_others = sum_others / (k - 1)
            # 클래스 임베딩과 음성 평균의 평균을 구함
            mean_common = (topk_embs + mean_others) / 2
            # 수정된 임베딩 계산
            modified_embs = topk_embs + alpha * mean_common - beta * mean_others

        elif method == 'qr_proj':
            # 배치 내에서만 동작하도록 tensor 크기를 B 단위로 변경
            topk_expanded = topk_embs.unsqueeze(2).expand(B, k, k, D)
            mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, k, k)
            others = topk_expanded[mask].view(B, k, k-1, D)
            # (B, k, k-1, D) -> (B, k, D, k-1) -> (B*k, D, k-1)
            others_t = others.transpose(2, 3)
            others_t_2d = others_t.reshape(B * k, D, (k - 1))
            Q_big, _ = torch.linalg.qr(others_t_2d.float(), mode='reduced')
            topk_flat = topk_embs.reshape(B * k, D)
            topk_flat_unsq = topk_flat.unsqueeze(-1).float()
            Q_t = Q_big.transpose(-1, -2)
            coefs = torch.bmm(Q_t, topk_flat_unsq)
            p = torch.bmm(Q_big, coefs)
            r = topk_flat_unsq - p
            e_j_new = alpha * r + beta * p
            e_j_new = e_j_new.squeeze(-1).reshape(B, k, D)
            modified_embs = e_j_new.half()

        elif method == 'svd_proj':
            topk_expanded = topk_embs.unsqueeze(2).expand(B, k, k, D)
            mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, k, k)
            others = topk_expanded[mask].view(B, k, k-1, D)
            others_t = others.transpose(2, 3)
            others_t_2d = others_t.reshape(B * k, D, (k - 1))
            U, _, _ = torch.linalg.svd(others_t_2d.float(), full_matrices=False)
            topk_flat = topk_embs.reshape(B * k, D)
            topk_flat_unsq = topk_flat.unsqueeze(-1)
            U_t = U.transpose(-1, -2)
            coefs = torch.bmm(U_t, topk_flat_unsq.float())
            p = torch.bmm(U, coefs.float())
            r = topk_flat_unsq - p
            e_j_new = alpha * r + beta * p
            e_j_new = e_j_new.squeeze(-1).reshape(B, k, D)
            modified_embs = e_j_new.half()

        elif method == 'svd_mm_proj':
            # 입력 검증: img_embs 필수
            if img_embs is None:
                raise ValueError("method='svd_mm_proj' requires `img_embs` to be provided.")
            
            # 현재 배치의 이미지 임베딩 추출 (B, D)
            img_embs_batch = img_embs[start:end].to(device)
            
            # 1. Negative 샘플 구성 (자기 자신 제외)
            # 원본 텐서 확장: (B, k, D) → (B, k, k, D)
            topk_expanded = topk_embs.unsqueeze(2).expand(B, k, k, D)
            
            # 마스크 생성: (k, k) eye 행렬 반전 → (B, k, k)
            mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, k, k)
            
            # 마스크 적용으로 자기 제외 이웃 추출 → (B, k, k-1, D)
            negative_others = topk_expanded[mask].view(B, k, k-1, D)

            # 2. Positive 샘플 구성 (이미지 임베딩 추가)
            # topk + 이미지 임베딩 결합 → (B, k+1, D)
            topk_plus_image = torch.cat([topk_embs, img_embs_batch.unsqueeze(1)], dim=1)
            
            # 차원 확장 → (B, k, k+1, D)
            topk_plus_image = topk_plus_image.unsqueeze(1).expand(B, k, k+1, D)

            # 3. 차원 재구성 (SVD 연산 준비)
            negative_others_t = negative_others.transpose(2, 3)  # (B, k, D, k-1)
            neg_2d = negative_others_t.reshape(B * k, D, k - 1)  # (B*k, D, k-1)
            
            full_t = topk_plus_image.transpose(2, 3)  # (B, k, D, k+1)
            full_2d = full_t.reshape(B * k, D, k + 1)  # (B*k, D, k+1)

            # 4. 기준 임베딩 평탄화 (B*k, D)
            e_j_flat = topk_embs.reshape(B * k, D)
            e_j_flat_unsq = e_j_flat.unsqueeze(-1)  # (B*k, D, 1)

            # 5. Negative 방향 SVD 투영
            # SVD: neg_2d = U_neg @ S_neg @ V_neg^T
            U_neg, _, _ = torch.linalg.svd(neg_2d.float(), full_matrices=False)
            U_neg_t = U_neg.transpose(-1, -2)  # (B*k, k-1, D)
            
            # 투영 계수 계산: c = U^T * e_j
            coefs_neg = torch.bmm(U_neg_t, e_j_flat_unsq.float())  # (B*k, k-1, 1)
            
            # 음성 부분공간 투영: p_neg = U * c
            p_neg = torch.bmm(U_neg, coefs_neg.float())  # (B*k, D, 1)
            
            # 직교 성분 추출: e_j_orth = e_j - p_neg
            e_j_orth = e_j_flat_unsq - p_neg  # (B*k, D, 1)

            # 6. Positive 방향 SVD 투영
            U_full, _, _ = torch.linalg.svd(full_2d.float(), full_matrices=False)
            U_full_t = U_full.transpose(-1, -2)  # (B*k, k+1, D)
            
            coefs_full = torch.bmm(U_full_t, e_j_flat_unsq.float())  # (B*k, k+1, 1)
            p_full = torch.bmm(U_full, coefs_full.float())  # (B*k, D, 1)

            # 7. 최종 임베딩 계산
            # 가중 합: α*(음성 직교 성분) + β*(양성 투영 성분)
            e_j_sum = alpha * e_j_orth + beta * p_full  # (B*k, D, 1)
            
            # 차원 복원: (B, k, D)
            e_j_sum = e_j_sum.squeeze(-1).reshape(B, k, D)
            modified_embs = e_j_sum  # 수정된 임베딩

        elif method == 'geodesic_svd_mm_proj':
            # 이하 모든 연산은 단위 구면(unit sphere) 상에서의 지오데식 연산 가정
            topk_embs_batch = topk_embs.to(device, dtype=float)
            img_embs_batch = img_embs[start:end].to(device, dtype=float)
            
            # 차원 정보 추출: [B: 배치, k: 이웃 수, D: 임베딩 차원]
            B_current, k, D = topk_embs_batch.shape
            
            # 1. Negative 샘플 구성 (자기 자신 제외)
            # 수식: N_ij = {e_m | m ∈ topk, m ≠ j} ∀i,j
            topk_expanded = topk_embs_batch.unsqueeze(2).expand(B_current, k, k, D)
            mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B_current, k, k)
            negative_others = topk_expanded[mask].view(B_current, k, k-1, D)
            
            # 2. Positive 샘플 구성 (원본 이미지 임베딩 추가)
            # 수식: P_i = {e_j | j ∈ topk} ∪ {e_img_i}
            topk_plus_image = torch.cat([topk_embs_batch, img_embs_batch.unsqueeze(1)], dim=1)
            topk_plus_image = topk_plus_image.unsqueeze(1).expand(B_current, k, k+1, D)
            
            # 3. Tangent space 투영 준비
            # 로그 맵 수식: v = log_e(e') = (e' - (e·e')e) * arccos(e·e') / sqrt(1 - (e·e')²)
            neg_others_2d = negative_others.reshape(B_current * k, k-1, D)
            log_neg = log_map_batch(e_j_flat_unsq, neg_others_2d)  # [B*k, k-1, D]
            log_neg_t = log_neg.transpose(1, 2)  # [B*k, D, k-1]
            
            # 4. Negative 방향 주성분 추출 (SVD)
            # 수식: U_neg Σ_neg V_neg^T = SVD([v_1 | ... | v_{k-1}])
            U_neg, _, _ = torch.linalg.svd(log_neg_t, full_matrices=False)
            U_neg_t = U_neg.transpose(-1, -2)  # [B*k, k-1, D]
            
            # 5. Tangent 벡터 음성 부분 공간으로 투영
            # 수식: p_neg = U_neg U_neg^T e_j_tangent
            coefs_neg = torch.bmm(U_neg_t, e_j_tangent.transpose(1, 2))
            p_neg = torch.bmm(U_neg, coefs_neg).transpose(1, 2)
            e_j_orth_tangent = e_j_tangent - p_neg  # 음성 공간 직교 성분
            
            # 6. Positive 방향 주성분 추출 (SVD)
            log_full = log_map_batch(e_j_flat_unsq, full_2d_reshape)  # [B*k, k+1, D]
            log_full_t = log_full.transpose(1, 2)  # [B*k, D, k+1]
            U_full, _, _ = torch.linalg.svd(log_full_t, full_matrices=False)
            U_full_t = U_full.transpose(-1, -2)  # [B*k, k+1, D]
            
            # 7. Tangent 벡터 양성 부분 공간으로 투영
            # 수식: p_full = U_full U_full^T e_j_tangent
            coefs_full = torch.bmm(U_full_t, e_j_tangent.transpose(1, 2))
            p_full = torch.bmm(U_full, coefs_full).transpose(1, 2)
            
            # 8. 지오데릭 조정: α(분리 강도)와 β(응집 강도)로 가중합
            # 최종 탄젠트 벡터: v' = α*(e_j_orth) + β*(p_full)
            e_j_prime_tangent = alpha * e_j_orth_tangent + beta * p_full
            
            # 9. 지수 맵으로 매니폴드 복원
            # 수식: e' = exp_e(v') = cos(||v'||)e + sin(||v'||)v'/||v'||
            e_j_prime = exp_map_batch(e_j_flat_unsq, e_j_prime_tangent)
            e_j_prime = e_j_prime.squeeze(1)
            
            # 10. 정규화 및 최종 임베딩
            modified_embs = e_j_prime.reshape(B_current, k, D)
            modified_embs = (modified_embs / modified_embs.norm(dim=-1, keepdim=True)).half()
        elif method == 'custom_proj_svd':
            # topk_embs: (B, k, D) – 배치 내 각 이미지의 top-k 텍스트 임베딩
            
            # 1. 각 샘플에 대해 SVD를 수행하여 공통 common vector를 추출
            #    각 샘플의 임베딩 행렬 X ∈ ℝ^(k×D)에 대해 SVD: X = U S V^T
            #    여기서 V^T의 첫 번째 행(즉, Vh[:, 0, :])가 principal right singular vector가 됩니다.
            U, S, Vh = torch.linalg.svd(topk_embs.float(), full_matrices=False)  # Vh: (B, k, D)
            common = Vh[:, 0, :]          # (B, D), 각 샘플에 대한 공통 벡터
            common = common.unsqueeze(1)  # (B, 1, D)로 확장하여 브로드캐스트 처리

            # 2. 각 임베딩 eᵢ에서, common vector 방향 성분을 제거하여 orthogonal component rᵢ를 구함
            #    rᵢ = eᵢ - (eᵢ · common)*common
            dot = (topk_embs * common).sum(dim=-1, keepdim=True)  # (B, k, 1)
            proj = dot * common                                   # (B, k, D): 각 eᵢ의 common 방향 성분
            R = topk_embs - proj                                  # (B, k, D): 각 임베딩의 orthogonal component (r₁, …, rₖ)
            
            # 3. 각 샘플에 대해 모든 r의 합을 구함
            sum_R = R.sum(dim=1, keepdim=True)  # (B, 1, D)
            
            # 4. 각 임베딩에 대해 최종 조정:
            #    eᵢ_adjusted = 2*rᵢ + common - (r₁ + r₂ + ... + rₖ)
            adjusted = 2 * R + common - sum_R   # (B, k, D)
            
            # 5. (옵션) 각 벡터를 단위 벡터로 정규화
            adjusted = adjusted / adjusted.norm(dim=-1, keepdim=True)
            
            modified_embs = adjusted

        else:
            raise ValueError("Unknown method")

        # (method가 geodesic_svd_mm_proj인 경우 이미 cos_sim을 계산한 경우가 있고, 그 외는 여기서 계산)
        if method != 'geodesic_svd_mm_proj':
            cos_sim = torch.nn.functional.cosine_similarity(modified_embs, topk_embs, dim=-1).mean()

        # 배치 결과는 CPU로 옮겨서 저장합니다.
        modified_embs_list.append(modified_embs.cpu())
        cos_sim_list.append(cos_sim.item())

    # 모든 배치 결과를 하나로 합칩니다.
    modified_embs_full = torch.cat(modified_embs_list, dim=0).cuda()
    mean_cos_sim = sum(cos_sim_list) / len(cos_sim_list)
    return modified_embs_full, mean_cos_sim




# ----------------------------------------------------------------
# 5) Evaluate the new embeddings and collect top-5 predictions
# ----------------------------------------------------------------
def evaluate_modified_embeddings(
    dataloader,
    model,
    modified_embs,
    topk_indices,
    classnames,
    device='cuda'
):
    """
    - modified_embs: (N, k, D) from build_modified_text_embs
    - topk_indices: (N, k)
    - We'll assume dataloader iterates in the *same order* as the first pass.

    For each batch, we do:
       image_embs: (B, D)
       text_embs:  (B, k, D)  => slice from modified_embs using idx range [idx_counter : idx_counter+B]
       => broadcast mul => sum => (B, k)
    Then pick argmax among k. Compare to ground truth if you need accuracy.
    """
    model.eval()
    idx_counter = 0

    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating modified embeddings"):
            batch = maybe_dictionarize_batch(batch)
            images, labels = batch['images'].cuda(), batch['labels'].cuda()

            B = images.size(0)
            # 1) Compute image embeddings
            image_embs = model.encode_image(images)
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)  # (B, D)

            # 2) Gather the slice of text embeddings for these B images
            #    shape (B, k, D)
            text_emb_slice = modified_embs[idx_counter: idx_counter + B].to(device)

            # 3) Compute dot product => (B, k)
            # Broadcast multiply:
            #   image_embs: (B, D) -> (B, 1, D)
            #   text_emb_slice: (B, k, D)
            # so result => (B, k, D), sum over D => (B, k)
            logits = 100.0 * torch.sum(
                image_embs.unsqueeze(1) * text_emb_slice, dim=-1
            )

            # 4) Get top-5 indices among the k available classes
            top5_idx = torch.topk(logits, k=min(5, logits.size(1)), dim=1, largest=True, sorted=True).indices  # shape (B, 5)

            # 5) If we want to measure actual accuracy vs. ground truth:
            #    We know topk_indices[i][ top5_idx[i] ] are the class indices chosen.
            slice_topk = topk_indices[idx_counter: idx_counter + B]  # shape (B, k)
            chosen_cls_idx = slice_topk[torch.arange(B).unsqueeze(1), top5_idx]  # shape (B, 5)

            # 6) Convert indices to class names and compare with ground truth
            for local_b in range(B):
                true_cname = classnames[labels[local_b].item()]
                pred_cnames = [classnames[idx.item()] for idx in chosen_cls_idx[local_b]]

                if true_cname == pred_cnames[0]:  # Top-1 match
                    top1_correct += 1
                if true_cname in pred_cnames:  # Top-5 match
                    top5_correct += 1

            idx_counter += B
            total += B

    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total

    print(f"[Modified Embeddings] Top-1 Accuracy: {top1_acc:.2f}% | Top-5 Accuracy: {top5_acc:.2f}% on {total} images.")
    return top1_acc, top5_acc

def evaluate_modified_embeddings_vectorized(
    args,
    dataloader,
    model,
    modified_embs,    # (N, k, D)
    topk_indices,     # (N, k)
    class_embeddings, # (C, D)
    device='cuda',
    img_embs=None,
    all_labels=None
):
    model.eval()
    all_probs = []
    # 1. Compute all image embeddings in one pass

    # 2. Compute logits in batches to save memory
    batch_size = 8192  # 메모리에 맞게 조정
    num_batches = (len(img_embs) + batch_size - 1) // batch_size
    all_logits = []
    
    for i in tqdm(range(num_batches), desc="Computing logits"):
        start = i * batch_size
        end = min((i+1)*batch_size, len(img_embs))
        img_batch = img_embs[start:end]  # (B, D)
        mod_emb_batch = modified_embs[start:end]  # (B, k, D)
        
        # Batch matrix multiplication (B, 1, D) * (B, k, D) -> (B, k)
        logits = 100.0 * torch.einsum('bd,bkd->bk', img_batch, mod_emb_batch)
        all_logits.append(logits)
    
    all_logits = torch.cat(all_logits, dim=0)  # (N, k)
    
    # 3. Get predictions using top-k indices
    # Convert topk_indices to class labels
    topk_classes = topk_indices  # (N, k)
    
    # Get predicted class indices
    pred_indices = topk_classes[torch.arange(len(all_logits)).unsqueeze(-1), 
                                all_logits.argsort(dim=1, descending=True)]
    
    # Calculate accuracy
    top1 = (pred_indices[:, 0] == all_labels).float().mean().item() * 100
    top5 = (pred_indices[:, :5] == all_labels.unsqueeze(-1)).any(dim=1).float().mean().item() * 100
    
    print(f"Top-1 Accuracy: {top1:.2f}%, Top-5 Accuracy: {top5:.2f}%")
    return top1, top5

def main():
    args = parse_arguments()
    device = 'cuda'
    model, preprocess = clip.load(args.model, device=device)

    print(f"{len(openai_classnames)} classes, {len(openai_imagenet_template)} templates")

    dataset = getattr(datasets, args.dataset)(
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    dataloader = dataset.test_loader

    # Collect top-k predictions for each image (if not cached)
    topk_pred_path = f'./results/Imagewise_{args.dataset}_top{args.topk}_predictions_{args.model.replace("/", "_").replace("-", "_")}.npy'
    if not os.path.exists(topk_pred_path):
        zeroshot_weights = zeroshot_classifier(args, model, openai_classnames, openai_imagenet_template)
        topk_predictions = collect_topk_indices(
            dataloader, model, zeroshot_weights, 
            k=args.topk, device=device
        )
        os.makedirs('./results', exist_ok=True)
        np.save(topk_pred_path, topk_predictions.cpu().numpy())
    else:
        topk_predictions = torch.from_numpy(np.load(topk_pred_path)).cuda()

    if os.path.exists(f'./checkpoints/{args.dataset}_{args.model.replace("/", "_").replace("-", "_")}_img_embs.pt'):
        img_embs = torch.load(f'./checkpoints/{args.dataset}_{args.model.replace("/", "_").replace("-", "_")}_img_embs.pt', map_location='cuda', weights_only=True)
        all_labels = torch.load(f'./checkpoints/{args.dataset}_{args.model.replace("/", "_").replace("-", "_")}_Labels.pt', map_location='cuda', weights_only=True)
    else:
        img_embs = []
        labels_list = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing image embeddings"):
                batch = maybe_dictionarize_batch(batch)
                images = batch['images'].to(device)
                labels = batch['labels'].to(device)
                img_emb = model.encode_image(images)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                img_embs.append(img_emb)
                labels_list.append(labels)
        img_embs = torch.cat(img_embs, dim=0).to(device)  # (N, D)
        all_labels = torch.cat(labels_list, dim=0).to(device)
        torch.save(img_embs, f'./checkpoints/{args.dataset}_{args.model.replace("/", "_").replace("-", "_")}_img_embs.pt')
        torch.save(all_labels, f'./checkpoints/{args.dataset}_{args.model.replace("/", "_").replace("-", "_")}_Labels.pt')
    

    zeroshot_weights = zeroshot_classifier(args, model, openai_classnames, openai_imagenet_template)
    class_embeddings = zeroshot_weights.t().cuda()  # (C, D)

    for alpha in [1.0,]:
        for beta in [0]:
            for method in ['geodesic_svd_mm_proj']:
                args.alpha = alpha
                args.beta = beta
                args.method = method
                modified_weights, modified_cos_sim = build_modified_text_embs(
                    class_embeddings=class_embeddings,
                    topk_indices=topk_predictions,
                    alpha=alpha, 
                    beta=beta,
                        method=args.method,
                    device='cuda',
                    img_embs=img_embs
                )
                print(f"[INFO] Alpha: {alpha}, Beta: {beta}, Method: {method}")
                print(f"[INFO] Modified Cosine Similarity: {modified_cos_sim}")
                print("------------------------------------------------------")

                top1_mod, top5_mod = evaluate_modified_embeddings_vectorized(
                    args,
                    dataloader,
                    model,
                    modified_weights,
                    topk_predictions,
                    class_embeddings=zeroshot_weights,
                    device='cuda',
                    img_embs=img_embs,
                    all_labels=all_labels
                )
                # ----------------------------------------------------------------
                # 6) (Optional) Log results with wandb
                # ----------------------------------------------------------------
                if not args.ignore_wandb:
                    wandb.init(entity="mlai_medical_ai", project="COE", config=args, reinit=True)
                    wandb.config.update(args)
                    run_name = f"{args.model.replace('/', '_').replace('-', '_')}_alpha_{args.alpha}_beta_{args.beta}"
                    wandb.run.name = run_name
                    wandb.log({
                        "Modified Top-1 accuracy": top1_mod,
                        "Modified Top-5 accuracy": top5_mod,
                        "Modified Cosine Similarity": modified_cos_sim
                    })

if __name__ == "__main__":
    main()