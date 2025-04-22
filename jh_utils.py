import numpy as np
import torch
import clip
from tqdm import tqdm
from imagenet_classnames import openai_classnames
from openai_imagenet_template import openai_imagenet_template
import datasets
import argparse
import os
from utils import maybe_dictionarize_batch, log_map_batch, exp_map_batch
from collections import Counter
import torch.nn.functional as F



#########################################
# 1. 수정된 벡터화된 텍스트 임베딩 생성 함수
#########################################

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

def pairwise_cosine_similarity(embeddings):
    """
    embeddings: (k, D) – 각 이미지의 후보 임베딩 집합
    반환: (k, k) cosine similarity 행렬 (대각원소는 1)
    """
    # 단위벡터 정규화
    embeddings_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
    # cosine similarity 행렬은 내적을 통해 얻음
    return embeddings_norm @ embeddings_norm.t()

def avg_offdiagonal_cos_sim(sim_matrix):
    """
    sim_matrix: (k, k) – cosine similarity 행렬
    대각원소(자기 자신과의 유사도)는 제외하고 평균 계산.
    """
    k = sim_matrix.shape[0]
    mask = ~torch.eye(k, dtype=torch.bool, device=sim_matrix.device)
    off_diag = sim_matrix[mask]
    return off_diag.mean().item()

def validate_adjustment_effectiveness(original_topk_embs, modified_embs):
    """
    original_topk_embs: (N, k, D) – 조정 전 top-k 임베딩
    modified_embs: (N, k, D) – 조정 후 top-k 임베딩

    각 이미지별로 후보 간 평균 cosine similarity와
    각 후보별 원본 vs. 조정 임베딩의 cosine similarity를 계산합니다.
    """
    N, k, D = original_topk_embs.shape

    avg_pairwise_orig = []
    avg_pairwise_mod = []
    avg_candidate_preservation = []  # 각 후보별 원본 vs. 조정 cosine similarity
    for i in range(N):
        # 각 이미지에 대한 후보 임베딩 추출
        orig = original_topk_embs[i]  # (k, D)
        mod = modified_embs[i]        # (k, D)

        # 1. 후보들 간 pairwise cosine similarity 행렬 구하기
        orig_sim = pairwise_cosine_similarity(orig)
        mod_sim = pairwise_cosine_similarity(mod)

        avg_orig = avg_offdiagonal_cos_sim(orig_sim)
        avg_mod = avg_offdiagonal_cos_sim(mod_sim)

        avg_pairwise_orig.append(avg_orig)
        avg_pairwise_mod.append(avg_mod)

        # 2. 각 후보별 원본 vs. 조정 임베딩의 cosine similarity 계산
        # 두 텐서를 각각 정규화한 후, elementwise 내적
        orig_norm = orig / orig.norm(dim=-1, keepdim=True)
        mod_norm = mod / mod.norm(dim=-1, keepdim=True)
        candidate_sim = (orig_norm * mod_norm).sum(dim=-1)  # (k,)
        avg_candidate_preservation.append(candidate_sim.mean().item())

    mean_pairwise_orig = sum(avg_pairwise_orig) / len(avg_pairwise_orig)
    mean_pairwise_mod = sum(avg_pairwise_mod) / len(avg_pairwise_mod)
    mean_candidate_preservation = sum(avg_candidate_preservation) / len(avg_candidate_preservation)

    print(f"평균 후보 간 cosine similarity (조정 전): {mean_pairwise_orig:.4f}")
    print(f"평균 후보 간 cosine similarity (조정 후):  {mean_pairwise_mod:.4f}")
    print(f"원본 vs. 조정 후보 간 cosine similarity:  {mean_candidate_preservation:.4f}")

    # 추가적으로 이미지별 통계나, 특정 그룹(예: 진도, 리트리버, 말티즈)에 대해서 필터링해서 평가할 수도 있습니다.
    return mean_pairwise_orig, mean_pairwise_mod, mean_candidate_preservation

# def build_modified_text_embs(
#     class_embeddings,  # (C, D)
#     topk_indices,      # (N, k)
#     alpha=1.0,
#     beta=1.0,
#     method='simple',
#     device='cuda',
#     img_embs=None,
#     batch_size=2048   # 원하는 배치 크기를 지정
# ):
#     """
#     Vectorized version of building modified text embeddings (batched version)
#     """
#     N, k = topk_indices.shape
#     C, D = class_embeddings.shape

#     modified_embs_list = []  # 각 배치 결과를 저장
#     cos_sim_list = []        # 각 배치별 cosine similarity를 저장 (평균값으로 나중에 합산)

#     if method == 'none':
#         modified_embs = class_embeddings
#         modified_cos_sim = 1
#         return modified_embs, modified_cos_sim
#     # 만약 method 에 text가 있다면
#     elif 'text' in method:
#         # 모든 text embedding (class_embeddings)으로부터 직교 기저를 구함
#         # SVD를 통해 Vh_text의 행들이 text subspace의 직교 기저 역할을 하도록 함
#         U_text, S_text, Vh_text = torch.linalg.svd(class_embeddings.float(), full_matrices=False)  # Vh_text: (C, D)

#     if 'unknown' in method:
#         # --- 1. 텍스트 정보 제거 (Null Text Embedding 활용) ---
#         # null_text_emb: (D,) 텍스트에서 클래스 정보 제거한 embedding
#         null_text_emb = get_null_text_embedding(model, template, device)  # model: CLIP, template: 텍스트 템플릿 리스트
#         null_text_emb = null_text_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
    
#         T = (topk_embs.float() @ null_text_emb.transpose(-1, -2)) * null_text_emb  # (B, k, 1) * (1, 1, D) → (B, k, D)
    
#     # 전체 N개를 배치 단위로 순회합니다.
#     for start in tqdm(range(0, N, batch_size), desc="Processing batches"):
#         end = min(N, start + batch_size)
#         B = end - start  # 현재 배치 크기

#         # 현재 배치에 해당하는 topk_indices (shape: (B, k))
#         topk_indices_batch = topk_indices[start:end]
#         # 해당 배치의 top-k 텍스트 임베딩 추출 (shape: (B, k, D))
#         topk_embs = class_embeddings[topk_indices_batch.view(-1)].view(B, k, D).to(device)

#         if method == 'simple':
#             # 각 그룹별 평균 임베딩 계산 (B, 1, D)
#             sum_embs = torch.sum(topk_embs, dim=1, keepdim=True)
#             # 각 벡터를 제외한 나머지의 합 (B, k, D)
#             sum_others = sum_embs - topk_embs
#             mean_others = sum_others / (k - 1)
#             # 클래스 임베딩과 음성 평균의 평균을 구함
#             mean_common = (topk_embs + mean_others) / 2
#             # 수정된 임베딩 계산
#             modified_embs = topk_embs + alpha * mean_common - beta * mean_others

#         elif method == 'qr_proj':
#             # 배치 내에서만 동작하도록 tensor 크기를 B 단위로 변경
#             topk_expanded = topk_embs.unsqueeze(2).expand(B, k, k, D)
#             mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, k, k)
#             others = topk_expanded[mask].view(B, k, k-1, D)
#             # (B, k, k-1, D) -> (B, k, D, k-1) -> (B*k, D, k-1)
#             others_t = others.transpose(2, 3)
#             others_t_2d = others_t.reshape(B * k, D, (k - 1))
#             Q_big, _ = torch.linalg.qr(others_t_2d.float(), mode='reduced')
#             topk_flat = topk_embs.reshape(B * k, D)
#             topk_flat_unsq = topk_flat.unsqueeze(-1).float()
#             Q_t = Q_big.transpose(-1, -2)
#             coefs = torch.bmm(Q_t, topk_flat_unsq)
#             p = torch.bmm(Q_big, coefs)
#             r = topk_flat_unsq - p
#             e_j_new = alpha * r + beta * p
#             e_j_new = e_j_new.squeeze(-1).reshape(B, k, D)
#             modified_embs = e_j_new.half()

#         elif method == 'svd_proj':
#             topk_expanded = topk_embs.unsqueeze(2).expand(B, k, k, D)
#             mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, k, k)
#             others = topk_expanded[mask].view(B, k, k-1, D)
#             others_t = others.transpose(2, 3)
#             others_t_2d = others_t.reshape(B * k, D, (k - 1))
#             U, _, _ = torch.linalg.svd(others_t_2d.float(), full_matrices=False)
#             topk_flat = topk_embs.reshape(B * k, D)
#             topk_flat_unsq = topk_flat.unsqueeze(-1)
#             U_t = U.transpose(-1, -2)
#             coefs = torch.bmm(U_t, topk_flat_unsq.float())
#             p = torch.bmm(U, coefs.float())
#             r = topk_flat_unsq - p
#             e_j_new = alpha * r + beta * p
#             e_j_new = e_j_new.squeeze(-1).reshape(B, k, D)
#             modified_embs = e_j_new.half()

#         elif method == 'svd_mm_proj':
#             # 입력 검증: img_embs 필수
#             if img_embs is None:
#                 raise ValueError("method='svd_mm_proj' requires `img_embs` to be provided.")
            
#             # 현재 배치의 이미지 임베딩 추출 (B, D)
#             img_embs_batch = img_embs[start:end].to(device)
            
#             # 1. Negative 샘플 구성 (자기 자신 제외)
#             # 원본 텐서 확장: (B, k, D) → (B, k, k, D)
#             topk_expanded = topk_embs.unsqueeze(2).expand(B, k, k, D)
            
#             # 마스크 생성: (k, k) eye 행렬 반전 → (B, k, k)
#             mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, k, k)
            
#             # 마스크 적용으로 자기 제외 이웃 추출 → (B, k, k-1, D)
#             negative_others = topk_expanded[mask].view(B, k, k-1, D)

#             # 2. Positive 샘플 구성 (이미지 임베딩 추가)
#             # topk + 이미지 임베딩 결합 → (B, k+1, D)
#             topk_plus_image = torch.cat([topk_embs, img_embs_batch.unsqueeze(1)], dim=1)
            
#             # 차원 확장 → (B, k, k+1, D)
#             topk_plus_image = topk_plus_image.unsqueeze(1).expand(B, k, k+1, D)

#             # 3. 차원 재구성 (SVD 연산 준비)
#             negative_others_t = negative_others.transpose(2, 3)  # (B, k, D, k-1)
#             neg_2d = negative_others_t.reshape(B * k, D, k - 1)  # (B*k, D, k-1)
            
#             full_t = topk_plus_image.transpose(2, 3)  # (B, k, D, k+1)
#             full_2d = full_t.reshape(B * k, D, k + 1)  # (B*k, D, k+1)

#             # 4. 기준 임베딩 평탄화 (B*k, D)
#             e_j_flat = topk_embs.reshape(B * k, D)
#             e_j_flat_unsq = e_j_flat.unsqueeze(-1)  # (B*k, D, 1)

#             # 5. Negative 방향 SVD 투영
#             # SVD: neg_2d = U_neg @ S_neg @ V_neg^T
#             U_neg, _, _ = torch.linalg.svd(neg_2d.float(), full_matrices=False)
#             U_neg_t = U_neg.transpose(-1, -2)  # (B*k, k-1, D)
            
#             # 투영 계수 계산: c = U^T * e_j
#             coefs_neg = torch.bmm(U_neg_t, e_j_flat_unsq.float())  # (B*k, k-1, 1)
            
#             # 음성 부분공간 투영: p_neg = U * c
#             p_neg = torch.bmm(U_neg, coefs_neg.float())  # (B*k, D, 1)
            
#             # 직교 성분 추출: e_j_orth = e_j - p_neg
#             e_j_orth = e_j_flat_unsq - p_neg  # (B*k, D, 1)

#             # 6. Positive 방향 SVD 투영
#             U_full, _, _ = torch.linalg.svd(full_2d.float(), full_matrices=False)
#             U_full_t = U_full.transpose(-1, -2)  # (B*k, k+1, D)
            
#             coefs_full = torch.bmm(U_full_t, e_j_flat_unsq.float())  # (B*k, k+1, 1)
#             p_full = torch.bmm(U_full, coefs_full.float())  # (B*k, D, 1)

#             # 7. 최종 임베딩 계산
#             # 가중 합: α*(음성 직교 성분) + β*(양성 투영 성분)
#             e_j_sum = alpha * e_j_orth + beta * p_full  # (B*k, D, 1)
            
#             # 차원 복원: (B, k, D)
#             e_j_sum = e_j_sum.squeeze(-1).reshape(B, k, D)
#             modified_embs = e_j_sum  # 수정된 임베딩

#         elif method == 'custom_proj_svd':
#             # topk_embs: (B, k, D) – 배치 내 각 이미지의 top-k 텍스트 임베딩
            
#             # 1. 각 샘플에 대해 SVD를 수행하여 공통 common vector를 추출
#             #    각 샘플의 임베딩 행렬 X ∈ ℝ^(k×D)에 대해 SVD: X = U S V^T
#             #    여기서 V^T의 첫 번째 행(즉, Vh[:, 0, :])가 principal right singular vector가 됩니다.
#             U, S, Vh = torch.linalg.svd(topk_embs.float(), full_matrices=False)  # Vh: (B, k, D)
#             common = Vh[:, 0, :]          # (B, D), 각 샘플에 대한 공통 벡터
#             common = common.unsqueeze(1)  # (B, 1, D)로 확장하여 브로드캐스트 처리

#             # 2. 각 임베딩 eᵢ에서, common vector 방향 성분을 제거하여 orthogonal component rᵢ를 구함
#             #    rᵢ = eᵢ - (eᵢ · common)*common
#             dot = (topk_embs * common).sum(dim=-1, keepdim=True)  # (B, k, 1)
#             proj = dot * common                                   # (B, k, D): 각 eᵢ의 common 방향 성분
#             R = topk_embs - proj                                  # (B, k, D): 각 임베딩의 orthogonal component (r₁, …, rₖ)
            
#             # 3. 각 샘플에 대해 모든 r의 합을 구함
#             sum_R = R.sum(dim=1, keepdim=True)  # (B, 1, D)
            
#             # 4. 각 임베딩에 대해 최종 조정:
#             #    eᵢ_adjusted = 2*rᵢ + common - (r₁ + r₂ + ... + rₖ)
#             adjusted = 2 * R + common - sum_R   # (B, k, D)
            
#             # 5. (옵션) 각 벡터를 단위 벡터로 정규화
#             adjusted = adjusted / adjusted.norm(dim=-1, keepdim=True)
            
#             modified_embs = adjusted

#         elif method == 'text_svd_proj':
#             # --- 1. 텍스트 정보 제거 ---
#             # 각 top-k embedding에 대해, 전체 text subspace (Vh_text)를 이용해 사영
#             # T = (v @ Vh_text^T) @ Vh_text
#             T = (topk_embs.float() @ Vh_text.T) @ Vh_text  # (B, k, D)
#             # sanity check, T가 random일때도 같은 성능일지
#             # T = torch.randn_like(T).cuda()
#             topk_embs_no_text = topk_embs.float() - T  # 텍스트 정보가 제거된 embedding

#             # --- 2. SVD Projection을 통한 top-k 간 orthogonalize ---
#             topk_expanded = topk_embs_no_text.unsqueeze(2).expand(B, k, k, D)
#             mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, k, k)
#             others = topk_expanded[mask].view(B, k, k-1, D)
#             others_t = others.transpose(2, 3)
#             others_t_2d = others_t.reshape(B * k, D, (k - 1))
#             U, _, _ = torch.linalg.svd(others_t_2d, full_matrices=False)
#             topk_flat = topk_embs_no_text.reshape(B * k, D)
#             topk_flat_unsq = topk_flat.unsqueeze(-1)
#             U_t = U.transpose(-1, -2)
#             coefs = torch.bmm(U_t, topk_flat_unsq)
#             p = torch.bmm(U, coefs)
#             r = topk_flat_unsq - p
#             e_j_new = alpha * r + beta * p
#             e_j_new = e_j_new.squeeze(-1).reshape(B, k, D)

#             # --- 3. 텍스트 정보 복원 ---
#             modified_embs = e_j_new + T
#             modified_embs = modified_embs / modified_embs.norm(dim=-1, keepdim=True)
#             modified_embs = modified_embs.half()

#         elif method == 'text_unknown_svd_proj':
#             # --- 1. 텍스트 정보 제거 ---
            
            

#             # --- 2. SVD Projection을 통한 top-k 간 orthogonalize ---
#             topk_expanded = topk_embs_no_text.unsqueeze(2).expand(B, k, k, D)
#             mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, k, k)
#             others = topk_expanded[mask].view(B, k, k-1, D)
#             others_t = others.transpose(2, 3)
#             others_t_2d = others_t.reshape(B * k, D, (k - 1))
#             U, _, _ = torch.linalg.svd(others_t_2d, full_matrices=False)
#             topk_flat = topk_embs_no_text.reshape(B * k, D)
#             topk_flat_unsq = topk_flat.unsqueeze(-1)
#             U_t = U.transpose(-1, -2)
#             coefs = torch.bmm(U_t, topk_flat_unsq)
#             p = torch.bmm(U, coefs)
#             r = topk_flat_unsq - p
#             e_j_new = alpha * r + beta * p
#             e_j_new = e_j_new.squeeze(-1).reshape(B, k, D)

#             # --- 3. 텍스트 정보 복원 ---
#             modified_embs = e_j_new + T
#             modified_embs = modified_embs / modified_embs.norm(dim=-1, keepdim=True)
#             modified_embs = modified_embs.half()

#         else:
#             raise ValueError("Unknown method")

#         # (method가 geodesic_svd_mm_proj인 경우 이미 cos_sim을 계산한 경우가 있고, 그 외는 여기서 계산)
#         if method != 'geodesic_svd_mm_proj':
#             cos_sim = torch.nn.functional.cosine_similarity(modified_embs, topk_embs, dim=-1).mean()

#         # 배치 결과는 CPU로 옮겨서 저장합니다.
#         modified_embs_list.append(modified_embs.cpu())
#         cos_sim_list.append(cos_sim.item())

#     # 모든 배치 결과를 하나로 합칩니다.
#     modified_embs_full = torch.cat(modified_embs_list, dim=0).cuda()
#     mean_cos_sim = sum(cos_sim_list) / len(cos_sim_list)
#     return modified_embs_full, mean_cos_sim

def build_modified_text_embs_vectorized(
    args,
    class_embeddings,  # (C, D)
    topk_indices,      # (N, k)
    alpha=1.0,
    beta=1.0,
    method='simple',
    device='cuda',
    img_embs=None,
    null_text_emb=None,
    batch_size=2048   # 원하는 배치 크기
):
    """
    배치별로 수정된 텍스트 임베딩(modified text embeddings)을 생성하는 벡터화된 버전.
    
    추가로, 각 이미지에 대해 아래와 같이 여러 종류의 embedding을 분리합니다.
    
    [simple method]
      - feature embedding: 원래 top-k 후보들의 평균 (즉, topk_embs.mean(dim=1))
      - negative embedding: 각 후보에서 나머지 후보들의 평균 (mean_others.mean(dim=1))
      - group embedding: (topk_embs + mean_others)/2를 후보별로 구한 후 평균 (mean_common.mean(dim=1))
      
    [projection 계열 (qr_proj, svd_proj, svd_mm_proj, geodesic_svd_mm_proj)]
      - raw embedding: 조정(adjust)되기 전의 top-k 후보 embedding (shape: (B, k, D))
      - projection embedding: 각 후보에서 구한 투영(p) 성분 (shape: (B, k, D))
      - orthogonal embedding: 각 후보에서 구한 orthogonal (r) 성분 (shape: (B, k, D))
      - group embedding: 여전히 후보들에 대해 p의 평균 (p.mean(dim=1))
      
    반환:
      - modified_embs_full: (N, k, D) 수정된 텍스트 임베딩들 (원래 코드에서 사용하는 용도)
      - group_embs_full: (N, D) 기본 group embedding (simple: mean_common 평균, projection: p의 평균)
      - mean_cos_sim: 전체 배치에서의 평균 코사인 유사도
      - extra_embeds: dict  
            * simple method: {"feature": feature_embeds_full, "negative": negative_embeds_full, "group": group_embeds_full}
            * projection 계열: {"raw": raw_embeds_full, "orthogonal": orthogonal_embeds_full, "projection": projection_embeds_full}
    """
    N, k = topk_indices.shape
    C, D = class_embeddings.shape

    cos_sim_list = []         # scalar per batch
    modified_embs_list = []

    if method == 'none':
        modified_embs = class_embeddings
        modified_cos_sim = 1
        return modified_embs, modified_cos_sim
    # 만약 method 에 text가 있다면
    elif 'text' in method:
        # 모든 text embedding (class_embeddings)으로부터 직교 기저를 구함
        # SVD를 통해 Vh_text의 행들이 text subspace의 직교 기저 역할을 하도록 함
        U_text, S_text, Vh_text = torch.linalg.svd(class_embeddings.float(), full_matrices=False)  # Vh_text: (C, D)

    # extra embedding들을 배치별로 모으기 위한 리스트
    if method == 'simple':
        extra_feature_list = []
        extra_negative_list = []
        extra_group_list = []
    elif 'proj' in method:
        extra_raw_list = []  # 조정 전 top-k embedding
        extra_orthogonal_list = []
        extra_projection_list = []

    for start in tqdm(range(0, N, batch_size), desc="Processing batches"):
        end = min(N, start + batch_size)
        B = end - start  # 현재 배치 크기

        # 배치에 해당하는 top-k 인덱스를 이용해 원본 텍스트 임베딩 (B, k, D)
        topk_indices_batch = topk_indices[start:end]
        topk_embs = class_embeddings[topk_indices_batch.view(-1)].view(B, k, D).to(device)
        
        # 원본 top-k embedding은 어떤 방법이든 비교용으로 저장
        if 'proj' in method:
            extra_raw_list.append(topk_embs.cpu())

        if method == 'simple':
            # 각 이미지에 대해 후보 임베딩의 합과 나머지 평균 계산
            sum_embs = torch.sum(topk_embs, dim=1, keepdim=True)          # (B, 1, D)
            sum_others = sum_embs - topk_embs                                # (B, k, D)
            mean_others = sum_embs - topk_embs                                # (B, k, D) --> (실제로 sum_embs - topk_embs)
            mean_others = sum_others / (k - 1)                               # (B, k, D)
            mean_common = (topk_embs + mean_others) / 2                      # (B, k, D)
            # 수정된 임베딩 계산
            modified_embs = topk_embs + alpha * mean_common - beta * mean_others
            # group embedding: 각 후보별 mean_common의 평균
            group_embs = mean_common.mean(dim=1)                             # (B, D)
            # extra embedding: feature = topk_embs 평균, negative = mean_others 평균
            feature_embs = topk_embs.mean(dim=1)                             # (B, D)
            negative_embs = mean_others.mean(dim=1)                          # (B, D)

            extra_feature_list.append(feature_embs.cpu())
            extra_negative_list.append(negative_embs.cpu())
            extra_group_list.append(group_embs.cpu())

        elif method == 'qr_proj':
            topk_expanded = topk_embs.unsqueeze(2).expand(B, k, k, D)
            mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, k, k)
            others = topk_expanded[mask].view(B, k, k-1, D)
            others_t = others.transpose(2, 3)
            others_t_2d = others_t.reshape(B * k, D, (k - 1))
            Q_big, _ = torch.linalg.qr(others_t_2d.float(), mode='reduced')
            topk_flat = topk_embs.reshape(B * k, D)
            topk_flat_unsq = topk_flat.unsqueeze(-1).float()
            Q_t = Q_big.transpose(-1, -2)
            coefs = torch.bmm(Q_t, topk_flat_unsq)
            p = torch.bmm(Q_big, coefs)     # (B*k, D, 1): projection 성분
            r = topk_flat_unsq - p          # (B*k, D, 1): orthogonal 성분
            e_j_new = alpha * r + beta * p
            e_j_new = e_j_new.squeeze(-1).reshape(B, k, D)
            modified_embs = e_j_new.half()
            # group embedding: p의 평균 (계속 평균 사용)
            p_reshaped = p.squeeze(-1).reshape(B, k, D)
            group_embs = p_reshaped.mean(dim=1)
            # extra embedding: raw는 already saved; orthogonal 및 projection은 그대로 보존
            r_reshaped = r.squeeze(-1).reshape(B, k, D)
            extra_projection_list.append(p_reshaped.cpu())
            extra_orthogonal_list.append(r_reshaped.cpu())

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
            p_reshaped = p.squeeze(-1).reshape(B, k, D)
            group_embs = p_reshaped.mean(dim=1)
            r_reshaped = r.squeeze(-1).reshape(B, k, D)
            extra_projection_list.append(p_reshaped.cpu())
            extra_orthogonal_list.append(r_reshaped.cpu())

        elif method == 'svd_mm_proj':
            if img_embs is None:
                raise ValueError("method='svd_mm_proj' requires `img_embs` to be provided.")
            # 이미지 임베딩 배치 추출 (B, D)
            img_embs_batch = img_embs[start:end].to(device)
            
            # --- SVD 공간1: top-k 후보에서 자기 자신을 제외한 others ---
            # topk_embs: (B, k, D)
            # 각 이미지별로, 각 후보에 대해 나머지 후보들(자기 자신 제외)을 모읍니다.
            topk_expanded = topk_embs.unsqueeze(2).expand(B, k, k, D)            # (B, k, k, D)
            mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, k, k)  # (B, k, k)
            others = topk_expanded[mask].view(B, k, k-1, D)                       # (B, k, k-1, D)
            others_t = others.transpose(2, 3)                                     # (B, k, D, k-1)
            # reshape: 각 후보에 대해, others: (D, k-1)
            others_t_2d = others_t.reshape(B * k, D, k - 1)
            # SVD 공간1 계산
            U1, _, _ = torch.linalg.svd(others_t_2d.float(), full_matrices=False)
            # 각 candidate vector (B*k, D, 1)
            candidate_flat = topk_embs.reshape(B * k, D)
            candidate_unsq = candidate_flat.unsqueeze(-1)                        # (B*k, D, 1)
            # 투영: p1 = U1 * (U1^T * candidate)
            U1_t = U1.transpose(-1, -2)
            coefs1 = torch.bmm(U1_t, candidate_unsq.float())
            p1 = torch.bmm(U1, coefs1.float())                                    # (B*k, D, 1)
            # orthogonal vector (SVD1 결과)
            r1 = candidate_unsq - p1                                              # (B*k, D, 1)
            # --- SVD 공간2: top-k 전체와 이미지 임베딩을 추가한 공간 ---
            # 각 이미지별로, top-k 후보 (shape: (k, D))와 이미지 임베딩 (shape: (1, D))을 concat → (k+1, D)
            topk_plus_image = torch.cat([topk_embs, img_embs_batch.unsqueeze(1)], dim=1)  # (B, k+1, D)
            # 이 공간을 모든 후보에 대해 동일하게 사용하기 위해, expand하여 (B, k, k+1, D)
            topk_plus_image_exp = topk_plus_image.unsqueeze(1).expand(B, k, k+1, D)
            # transpose: (B, k, D, k+1)
            full_t = topk_plus_image_exp.transpose(2, 3)
            # reshape: (B*k, D, k+1)
            full_2d = full_t.reshape(B * k, D, k + 1)
            # SVD 공간2 계산
            U2, _, _ = torch.linalg.svd(full_2d.float(), full_matrices=False)
            U2_t = U2.transpose(-1, -2)
            coefs2 = torch.bmm(U2_t, candidate_unsq.float())
            p2 = torch.bmm(U2, coefs2.float())                                   # (B*k, D, 1)
            
            # --- 최종 벡터 결합 ---
            # 최종 candidate vector = α * (orthogonal vector from SVD1) + β * (projection vector from SVD2)
            final_candidate = alpha * r1 + beta * p2                               # (B*k, D, 1)
            final_candidate = final_candidate.squeeze(-1).reshape(B, k, D)          # (B, k, D)
            modified_embs = final_candidate  # 최종 조정된 후보 임베딩
            
            # group embedding: projection vector (SVD 공간2의 p2)의 평균 (각 이미지 내 후보별 평균)
            p2_reshaped = p2.squeeze(-1).reshape(B, k, D)
            group_embs = p2_reshaped.mean(dim=1)                                  # (B, D)
            
            # extra embeddings:
            # - raw: 원본 top-k embedding (이미 저장)
            # - orthogonal: SVD1 결과, r1 (reshape 후, (B, k, D))
            # - projection: SVD2 결과, p2 (reshape 후, (B, k, D))
            extra_orthogonal = r1.squeeze(-1).reshape(B, k, D)
            extra_projection = p2_reshaped
            # extra raw is handled outside (see below in extra_embeds dict)
            extra_projection_list.append(extra_projection.cpu())
            extra_orthogonal_list.append(extra_orthogonal.cpu())

        elif method == 'geodesic_svd_mm_proj':
            topk_embs_batch = topk_embs.to(device, dtype=float)
            img_embs_batch = img_embs[start:end].to(device, dtype=float)
            B_current, k, D = topk_embs_batch.shape
            topk_expanded = topk_embs_batch.unsqueeze(2).expand(B_current, k, k, D)
            mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B_current, k, k)
            negative_others = topk_expanded[mask].view(B_current, k, k-1, D)
            topk_plus_image = torch.cat([topk_embs_batch, img_embs_batch.unsqueeze(1)], dim=1)
            topk_plus_image = topk_plus_image.unsqueeze(1).expand(B_current, k, k+1, D)
            negative_others_t = negative_others.transpose(2, 3)
            neg_2d = negative_others_t.reshape(B_current * k, D, k-1)
            full_t = topk_plus_image.transpose(2, 3)
            full_2d = full_t.reshape(B_current * k, D, k+1)
            e_j_flat = topk_embs_batch.reshape(B_current * k, D)
            e_j_flat_unsq = e_j_flat.unsqueeze(1)
            # log_map_batch는 사용자가 정의한 함수라고 가정
            neg_others_2d = negative_others.reshape(B_current * k, k-1, D)
            log_neg = log_map_batch(e_j_flat_unsq, neg_others_2d)
            log_neg_t = log_neg.transpose(1, 2)
            U_neg, _, _ = torch.linalg.svd(log_neg_t, full_matrices=False)
            e_j_tangent = torch.zeros_like(e_j_flat_unsq)
            U_neg_t = U_neg.transpose(-1, -2)
            coefs_neg = torch.bmm(U_neg_t, e_j_tangent.transpose(1, 2))
            p_neg = torch.bmm(U_neg, coefs_neg).transpose(1, 2)
            e_j_orth_tangent = e_j_tangent - p_neg
            full_2d_reshape = topk_plus_image.reshape(B_current * k, k+1, D)
            log_full = log_map_batch(e_j_flat_unsq, full_2d_reshape)
            log_full_t = log_full.transpose(1, 2)
            U_full, _, _ = torch.linalg.svd(log_full_t, full_matrices=False)
            U_full_t = U_full.transpose(-1, -2)
            coefs_full = torch.bmm(U_full_t, e_j_tangent.transpose(1, 2))
            p_full = torch.bmm(U_full, coefs_full).transpose(1, 2)
            e_j_prime_tangent = alpha * e_j_orth_tangent + beta * p_full
            e_j_prime = exp_map_batch(e_j_flat_unsq, e_j_prime_tangent)
            e_j_prime = e_j_prime.squeeze(1)
            modified_embs = e_j_prime.reshape(B_current, k, D)
            modified_embs = (modified_embs / modified_embs.norm(dim=-1, keepdim=True)).half()
            p_full_reshaped = p_full.reshape(B_current, k, D)
            group_embs = p_full_reshaped.mean(dim=1)
            r_reshaped = (e_j_flat_unsq - p_full).squeeze(1).reshape(B_current, k, D)
            extra_projection_list.append(p_full_reshaped.cpu())
            extra_orthogonal_list.append(r_reshaped.cpu())
            cos_sim = F.cosine_similarity(modified_embs, topk_embs_batch, dim=-1).mean()
        elif method == 'custom_proj_svd':
            # topk_embs: (B, k, D) – 배치 내 각 이미지의 top-k 텍스트 임베딩
            B_current, k, D = topk_embs.shape
            
            # 1. 각 샘플에 대해 SVD를 수행하여 공통 common vector를 추출
            #    각 샘플의 임베딩 행렬 X ∈ ℝ^(k×D)에 대해 SVD: X = U S V^T
            #    여기서 V^T의 첫 번째 행(즉, Vh[:, 0, :])가 principal right singular vector가 됩니다.
            U, S, Vh = torch.linalg.svd(topk_embs.float(), full_matrices=False)  # Vh: (B, k, D)
            group_embs = Vh[:, 0, :]          # (B, D), 각 샘플에 대한 공통 벡터
            group_embs = group_embs.unsqueeze(1)  # (B, 1, D)로 확장하여 브로드캐스트 처리

            # 2. 각 임베딩 eᵢ에서, common vector 방향 성분을 제거하여 orthogonal component rᵢ를 구함
            #    rᵢ = eᵢ - (eᵢ · common)*common
            dot = (topk_embs * group_embs).sum(dim=-1, keepdim=True)  # (B, k, 1)
            proj = dot * group_embs                                   # (B, k, D): 각 eᵢ의 common 방향 성분
            R = topk_embs - proj                                  # (B, k, D): 각 임베딩의 orthogonal component (r₁, …, rₖ)
            
            # 3. 각 샘플에 대해 모든 r의 합을 구함
            sum_R = R.sum(dim=1, keepdim=True)  # (B, 1, D)
            
            # 4. 각 임베딩에 대해 최종 조정:
            #    eᵢ_adjusted = 2*rᵢ + common - (r₁ + r₂ + ... + rₖ)
            adjusted = 2 * R + group_embs - sum_R   # (B, k, D)
            
            # 5. (옵션) 각 벡터를 단위 벡터로 정규화
            adjusted = adjusted / adjusted.norm(dim=-1, keepdim=True)
            
            # (B, K, D) 형태로 확장하여 extra_projection_list에 저장
            group_embs_reshaped = group_embs.expand(B_current, k, D)
            extra_projection_list.append(group_embs_reshaped.cpu())

            # extra_orthogonal_list도 저장
            extra_orthogonal_list.append(R.cpu())
            modified_embs = adjusted
            
        # elif method == 'text_svd_proj':
        #     # --- 1. 텍스트 정보 제거 ---
        #     # 각 top-k embedding에 대해, 전체 text subspace (Vh_text)를 이용해 사영
        #     # T = (v @ Vh_text^T) @ Vh_text
        #     T = (topk_embs.float() @ Vh_text.T) @ Vh_text  # (B, k, D)
        #     # sanity check, T가 random일때도 같은 성능일지
        #     # T = torch.randn_like(T).cuda()
        #     topk_embs_no_text = topk_embs.float() - T  # 텍스트 정보가 제거된 embedding

        #     # --- 2. SVD Projection을 통한 top-k 간 orthogonalize ---
        #     topk_expanded = topk_embs_no_text.unsqueeze(2).expand(B, k, k, D)
        #     mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, k, k)
        #     others = topk_expanded[mask].view(B, k, k-1, D)
        #     others_t = others.transpose(2, 3)
        #     others_t_2d = others_t.reshape(B * k, D, (k - 1))
        #     U, _, _ = torch.linalg.svd(others_t_2d, full_matrices=False)
        #     topk_flat = topk_embs_no_text.reshape(B * k, D)
        #     topk_flat_unsq = topk_flat.unsqueeze(-1)
        #     U_t = U.transpose(-1, -2)
        #     coefs = torch.bmm(U_t, topk_flat_unsq)
        #     p = torch.bmm(U, coefs)
        #     r = topk_flat_unsq - p
        #     e_j_new = alpha * r + beta * p
        #     e_j_new = e_j_new.squeeze(-1).reshape(B, k, D)

        #     # --- 3. 텍스트 정보 복원 ---
        #     modified_embs = e_j_new + T
        #     modified_embs = modified_embs / modified_embs.norm(dim=-1, keepdim=True)
        #     modified_embs = modified_embs.half()
            
        #     extra_projection_list.append(p.cpu())
        #     extra_orthogonal_list.append(r.cpu())
        # hyperparameters for this method (실험적으로 조정)

        elif method == 'text_svd_proj':
            # top_l = 3         # 제거할 주성분의 개수 (예: 상위 3개 또는 하위 3개)
            # mean_center = True  # True면 global mean을 빼고, 나중에 복원함
            # pc_mode = 'upper'   # 'upper'이면 상위 주성분, 'lower'이면 하위 주성분 사용
            # --- 0. Global SVD는 이미 수행되어 Vh_text (전체 텍스트 subspace의 직교 기저) 가 있음 ---
            # class_embeddings: (C, D)에서 SVD 수행 → Vh_text: (D, D) 혹은 (C, D) depending on shape

            # --- 1. 텍스트 정보 제거: mean-centering 및 선택적 주성분 제거 ---
            if args.mean_center:
                global_mean = class_embeddings.mean(dim=0, keepdim=True)  # (1, D)
                # center top-k embeddings
                topk_embs_centered = topk_embs.float() - global_mean  # (B, k, D)
            else:
                topk_embs_centered = topk_embs.float()

            # 선택적 주성분 선택: 'upper'이면 상위 top_l, 'lower'이면 하위 top_l
            if args.top_l is None or args.top_l <= 0:
                components = Vh_text  # 전체 주성분 사용 (기존 버전과 동일)
            elif args.pc_mode == 'upper':
                components = Vh_text[:args.top_l]  # (top_l, D)
            elif args.pc_mode == 'lower':
                components = Vh_text[-args.top_l:]  # (top_l, D)
            else:
                components = Vh_text[:args.top_l]  # 기본은 상위 주성분

            # 각 top-k 임베딩에 대해, 선택한 주성분 서브스페이스로의 투영
            # T = (v @ components.T) @ components, shape: (B, k, D)
            T = (topk_embs_centered @ components.T) @ components

            # 텍스트 공통 정보 제거: 원래 벡터에서 투영한 성분을 빼기
            topk_embs_no_text = topk_embs_centered - T

            # 만약 mean_centering을 했다면, 원래 중심(global_mean)을 복원
            if args.mean_center:
                topk_embs_no_text = topk_embs_no_text + global_mean

            # --- 2. SVD Projection을 통한 top-k 간 orthogonalize ---
            topk_expanded = topk_embs_no_text.unsqueeze(2).expand(B, k, k, D)
            mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, k, k)
            others = topk_expanded[mask].view(B, k, k-1, D)
            others_t = others.transpose(2, 3)
            others_t_2d = others_t.reshape(B * k, D, (k - 1))
            U, _, _ = torch.linalg.svd(others_t_2d, full_matrices=False)
            topk_flat = topk_embs_no_text.reshape(B * k, D)
            topk_flat_unsq = topk_flat.unsqueeze(-1)
            U_t = U.transpose(-1, -2)
            coefs = torch.bmm(U_t, topk_flat_unsq)
            p = torch.bmm(U, coefs)
            r = topk_flat_unsq - p
            e_j_new = alpha * r + beta * p
            e_j_new = e_j_new.squeeze(-1).reshape(B, k, D)

            # --- 3. 텍스트 정보 복원 ---
            modified_embs = e_j_new + T
            modified_embs = modified_embs / modified_embs.norm(dim=-1, keepdim=True)
            modified_embs = modified_embs.half()
            
            extra_projection_list.append(p.cpu())
            extra_orthogonal_list.append(r.cpu())

   
        elif method == 'text_svd_proj_unknown':
            
            # --- 1. 텍스트 정보 제거 ---
            T = null_text_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
               
            # 단순 뺄셈 ----    
            topk_embs_no_text = topk_embs.float() - T  # (B, k, D) - (1, 1, D)

            # --- 2. SVD Projection을 통한 top-k 간 orthogonalize ---
            topk_expanded = topk_embs_no_text.unsqueeze(2).expand(B, k, k, D)
            mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, k, k)
            others = topk_expanded[mask].view(B, k, k-1, D)
            others_t = others.transpose(2, 3)
            others_t_2d = others_t.reshape(B * k, D, (k - 1))
            U, _, _ = torch.linalg.svd(others_t_2d, full_matrices=False)
            topk_flat = topk_embs_no_text.reshape(B * k, D)
            topk_flat_unsq = topk_flat.unsqueeze(-1)
            U_t = U.transpose(-1, -2)
            coefs = torch.bmm(U_t, topk_flat_unsq)
            p = torch.bmm(U, coefs)
            r = topk_flat_unsq - p
            e_j_new = alpha * r + beta * p
            e_j_new = e_j_new.squeeze(-1).reshape(B, k, D)

            # --- 3. 텍스트 정보 복원 ---
            modified_embs = e_j_new + T
            modified_embs = modified_embs / modified_embs.norm(dim=-1, keepdim=True)
            modified_embs = modified_embs.half()
            
            extra_projection_list.append(p.cpu())
            extra_orthogonal_list.append(r.cpu())
            
        elif method == 'text_svd_proj_mm':
            # --- 1. 텍스트 정보 제거 ---
            if img_embs is None:
                raise ValueError("method='svd_mm_proj' requires `img_embs` to be provided.")
            # 이미지 임베딩 배치 추출 (B, D)
            img_embs_batch = img_embs[start:end].to(device)
            
            # 이미지 임베딩과 각 top-k 후보 간의 평균 임베딩 만들기 (= 텍스트 정보 제거)
            topk_embs_mm = (topk_embs + img_embs_batch.unsqueeze(1)) / 2  # (B, k, D)
            
            # --- SVD 공간1: top-k 후보에서 자기 자신을 제외한 others ---
            # topk_embs: (B, k, D)
            # 각 이미지별로, 각 후보에 대해 나머지 후보들(자기 자신 제외)을 모읍니다.
            topk_expanded = topk_embs_mm.unsqueeze(2).expand(B, k, k, D)            # (B, k, k, D)
            mask = (~torch.eye(k, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, k, k)  # (B, k, k)
            others = topk_expanded[mask].view(B, k, k-1, D)                       # (B, k, k-1, D)
            others_t = others.transpose(2, 3)                                     # (B, k, D, k-1)
            # reshape: 각 후보에 대해, others: (D, k-1)
            others_t_2d = others_t.reshape(B * k, D, k - 1)
            # SVD 공간1 계산
            U1, _, _ = torch.linalg.svd(others_t_2d.float(), full_matrices=False)
            # 각 candidate vector (B*k, D, 1)
            candidate = topk_embs_mm.reshape(B * k, D).unsqueeze(-1)   # (B*k, D, 1)
            # 투영: p1 = U1 * (U1^T * candidate)
            U1_t = U1.transpose(-1, -2)
            coefs1 = torch.bmm(U1_t, candidate.float())
            p = torch.bmm(U1, coefs1.float())                                    # (B*k, D, 1)
            # orthogonal vector (SVD1 결과)
            r = candidate - p                                              # (B*k, D, 1)
            
            # --- 최종 벡터 결합 ---
            # 최종 candidate vector = α * (orthogonal vector from SVD1) + β * (projection vector from SVD2)
            candidate = alpha * r + beta * p                               # (B*k, D, 1)
            candidate = candidate.squeeze(-1).reshape(B, k, D) + (topk_embs - topk_embs_mm) / 2         # (B, k, D)
            modified_embs = candidate  # 최종 조정된 후보 임베딩
            
            # extra embeddings:
            # - raw: 원본 top-k embedding (이미 저장)
            # - orthogonal: SVD1 결과, r1 (reshape 후, (B, k, D))
            # - projection: SVD2 결과, p2 (reshape 후, (B, k, D))
            extra_orthogonal = r.squeeze(-1).reshape(B, k, D)
            extra_projection = p.squeeze(-1).reshape(B, k, D)
            # extra raw is handled outside (see below in extra_embeds dict)
            extra_projection_list.append(extra_projection.cpu())
            extra_orthogonal_list.append(extra_orthogonal.cpu())
   
        else:
            raise ValueError("Unknown method")

        if method != 'geodesic_svd_mm_proj':
            cos_sim = F.cosine_similarity(modified_embs, topk_embs, dim=-1).mean()
        modified_embs_list.append(modified_embs.half().cpu())
        cos_sim_list.append(cos_sim.item())

    mean_cos_sim = sum(cos_sim_list) / len(cos_sim_list)
    # extra_embeds dict 생성
    if method == 'simple':
        extra_embeds = {
            "feature": torch.cat(extra_feature_list, dim=0).to(device),
            "negative": torch.cat(extra_negative_list, dim=0).to(device),
            "group": group_embs  # 이미 computed above
        }
    # proj 라는 단어가 있다면
    elif 'proj' in method:
        extra_embeds = {
            "raw": torch.cat(extra_raw_list, dim=0).half().to(device),  # 조정 전 top-k embedding
            "orthogonal": torch.cat(extra_orthogonal_list, dim=0).reshape(-1, k, D).half().to(device),
            "projection": torch.cat(extra_projection_list, dim=0).reshape(-1, k, D).half().to(device)
        }
    else:
        extra_embeds = None

    modified_embs = torch.cat(modified_embs_list, dim=0).to(device)

    return modified_embs, mean_cos_sim, extra_embeds

#########################################
# 2. 텍스트 임베딩 생성 함수 (이미 제공됨)
#########################################

def get_text_embedding(model, cls_list, template, device='cuda'):
    """
    cls_list 내의 클래스 이름에 대해 여러 템플릿(prompt)을 사용해 텍스트 임베딩의 평균값을 반환
    """
    texts = []
    for cname in cls_list:
        prompts = [t(cname) for t in template]
        texts.extend(prompts)
    tokenized = clip.tokenize(texts).to(device)
    with torch.no_grad():
        embs = model.encode_text(tokenized)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    mean_emb = embs.mean(dim=0)
    mean_emb = mean_emb / mean_emb.norm()
    return mean_emb

#########################################
# 3. 평가 함수: top-k embedding 각각에 대해 텍스트 임베딩과 유사도 계산
#########################################

def evaluate_topk_emb_similarity(embeddings, all_labels, classnames, target_group, custom_text, template, model, device='cuda'):
    """
    embeddings: (N, k, D) – 각 이미지의 top-k 후보 embedding
    ground-truth 클래스가 target_group에 해당하는 이미지들에 대해,
    각 이미지의 k개의 후보와 get_text_embedding()으로 얻은 custom_text (예:"dog") 텍스트 임베딩과의 코사인 유사도를 계산.
    
    최종적으로, 각 이미지의 k개 유사도를 모두 모아서 평균 유사도를 출력합니다.
    """
    selected_sims = []
    text_emb = get_text_embedding(model, [custom_text], template, device=device)  # (D,)
    text_emb = text_emb.unsqueeze(0)  # (1, D)
    for i in range(len(all_labels)):
        true_class = classnames[all_labels[i].item()]
        if true_class in target_group:
            # embeddings[i]: (k, D)
            candidate_sims = F.cosine_similarity(embeddings[i], text_emb.expand(embeddings[i].shape[0], -1), dim=1)
            selected_sims.append(candidate_sims)
    if len(selected_sims) == 0:
        print(f"No images found for group {target_group}.")
        return None
    all_sims = torch.cat(selected_sims, dim=0)
    avg_sim = all_sims.mean().item()
    print(f"Average cosine similarity for top-k candidates for {target_group} and '{custom_text}': {avg_sim:.4f}")
    return avg_sim

#########################################
# 4. 기존 평가 함수 (각 이미지당 하나의 embedding 평가)
#########################################

def evaluate_group_emb_similarity(embeddings, all_labels, classnames, target_group, custom_text, model, device='cuda'):
    """
    embeddings: (N, D) – 각 이미지에 해당하는 embedding
    ground-truth 클래스가 target_group에 해당하는 이미지들의 embedding들을 평균낸 후,
    get_text_embedding()으로 얻은 custom_text 텍스트 임베딩과의 코사인 유사도를 계산.
    """
    selected_embs = []
    for i in range(len(all_labels)):
        true_class = classnames[all_labels[i].item()]
        if true_class in target_group:
            selected_embs.append(embeddings[i])
    if len(selected_embs) == 0:
        print(f"No images found for group {target_group}.")
        return None
    aggregated_emb = torch.stack(selected_embs, dim=0).mean(dim=0)
    aggregated_emb = aggregated_emb / aggregated_emb.norm()
    text_emb = get_text_embedding(model, [custom_text], device=device)
    similarity = F.cosine_similarity(aggregated_emb.unsqueeze(0), text_emb.unsqueeze(0), dim=1).item()
    print(f"Cosine similarity between aggregated embedding for {target_group} and '{custom_text}' text: {similarity:.4f}")
    return similarity


def evaluate_raw_projection_cosine_similarity(raw_embeds, projection_embeds):
    """
    raw_embeds: (N, k, D) – 조정 전 top-k embedding
    projection_embeds: (N, k, D) – projection 벡터
    두 텐서의 각 대응되는 후보에 대해 코사인 유사도를 계산하고, 전체 평균 유사도를 출력합니다.
    """
    # 각 후보별로 코사인 유사도 계산 (마지막 차원 D 기준)
    cos_sims = F.cosine_similarity(raw_embeds, projection_embeds, dim=-1)  # 결과 shape: (N, k)
    avg_cos_sim = cos_sims.mean().item()
    print(f"Average cosine similarity between raw and projection vectors: {avg_cos_sim:.4f}, variance: {cos_sims.var().item():.4f}")
    return avg_cos_sim, cos_sims.var().item()

def load_text_checkpoint(args, model, template, device='cuda'):
    """
    model: CLIP 모델
    classnames: 클래스 이름 리스트
    template: 템플릿 리스트
    """
    checkpoint_path = os.path.join(args.model_location, f"{args.dataset}_{args.model.replace('/', '_').replace('-', '_')}_{args.custom_template}_text_embs.pt")
    if os.path.exists(checkpoint_path):
        print("[INFO] Loading zero-shot classifier from:", checkpoint_path)
        zeroshot_weights = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
    else:
        print("[INFO] Creating zero-shot classifier")
        zs_weights = []
        with torch.no_grad():
            for classname in openai_classnames:
                texts = [t(classname) for t in template]
                tokenized = clip.tokenize(texts).to(device)
                emb = model.encode_text(tokenized)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                zs_weights.append(emb.mean(dim=0) / emb.mean(dim=0).norm())
            zeroshot_weights = torch.stack(zs_weights, dim=1).to(device)
        os.makedirs('./checkpoints', exist_ok=True)
        torch.save(zeroshot_weights, checkpoint_path)
    return zeroshot_weights

def load_topk_predictions(args, model, dataloader, zeroshot_weights, device='cuda'):
    topk_pred_path = f'./results/Imagewise_{args.dataset}_top{args.topk}_predictions_{args.model.replace("/", "_").replace("-", "_")}.npy'
    if not os.path.exists(topk_pred_path):
        topk_list = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting top-k indices"):
                batch = maybe_dictionarize_batch(batch)
                images = batch['images'].to(device)
                img_embs = model.encode_image(images)
                img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
                logits = 100. * (img_embs @ zeroshot_weights)
                _, topk_idx = logits.topk(args.topk, dim=1)
                topk_list.append(topk_idx.cpu())
        topk_indices = torch.cat(topk_list, dim=0).to(device)
        os.makedirs('./results', exist_ok=True)
        np.save(topk_pred_path, topk_indices.cpu().numpy())
    else:
        topk_indices = torch.from_numpy(np.load(topk_pred_path)).to(device)
    return topk_indices

def load_image_embeddings(args, model, dataloader, device='cuda'):
    img_embs_path = f'./checkpoints/{args.dataset}_{args.model.replace("/", "_").replace("-", "_")}_img_embs.pt'
    labels_path = f'./checkpoints/{args.dataset}_{args.model.replace("/", "_").replace("-", "_")}_Labels.pt'
    if os.path.exists(img_embs_path) and os.path.exists(labels_path):
        img_embs = torch.load(img_embs_path, map_location=device, weights_only=True)
        all_labels = torch.load(labels_path, map_location=device, weights_only=True)
    else:
        img_embs_list = []
        labels_list = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing image embeddings"):
                batch = maybe_dictionarize_batch(batch)
                images = batch['images'].to(device)
                labels = batch['labels'].to(device)
                emb = model.encode_image(images)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                img_embs_list.append(emb)
                labels_list.append(labels)
        img_embs = torch.cat(img_embs_list, dim=0).to(device)
        all_labels = torch.cat(labels_list, dim=0).to(device)
        torch.save(img_embs, img_embs_path)
        torch.save(all_labels, labels_path)
    return img_embs, all_labels

def get_null_text_embedding(clip_model, template, device):
    """
    클래스 정보를 제거한 'null text embedding'을 계산한다.
    예: "a photo of <|endoftext|>", "a photo of something", "a photo of object" 등

    Args:
        clip_model: CLIP 모델 (encode_text 메서드를 가져야 함)
        template: 클래스 이름을 포함하는 text template 함수 리스트
        device: torch device

    Returns:
        null_text_emb: (D,) 크기의 평균 정규화된 텍스트 임베딩 (float32)
    """
    null_classnames = ["<|endoftext|>", "something", "object"]
    null_embs = []

    with torch.no_grad():
        for null_cls in null_classnames:
            texts = [t(null_cls) for t in template]  # 예: "a photo of object"
            tokenized = clip.tokenize(texts).to(device)
            emb = clip_model.encode_text(tokenized)  # (T, D)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # 정규화
            mean_emb = emb.mean(dim=0)
            mean_emb = mean_emb / mean_emb.norm()  # 다시 정규화
            null_embs.append(mean_emb)

    null_text_emb = torch.stack(null_embs, dim=0).mean(dim=0)
    null_text_emb = null_text_emb / null_text_emb.norm()  # 최종 정규화

    return null_text_emb.float()

    
    # 평가: ground-truth가 dog 관련 클래스(예: 'Labrador Retriever', 'Golden Retriever', 'Maltese')인 경우
    # dog_group = ["Labrador Retriever", "Golden Retriever", "Maltese"]
    # custom_text = "truck"

    # compare_classes = ["Granny Smith apple", "fig", "cucumber"]
    # compare_class_A = "Green fruit"
    # compare_class_B = "Red fruit"

    # if args.method == 'simple':
    #     for compare_class in compare_classes:
    #         print(f"=== Simple Method Evaluation for {compare_class} with '{compare_class_A}' label===")
    #         print("Feature Embedding:")
    #         evaluate_group_emb_similarity(extra_embeds["feature"], all_labels, openai_classnames, [compare_class], compare_class_A, model, device=device)
    #         print("Negative Embedding:")
    #         evaluate_group_emb_similarity(extra_embeds["negative"], all_labels, openai_classnames, [compare_class], compare_class_A, model, device=device)
    #         print("Group Embedding:")
    #         evaluate_group_emb_similarity(extra_embeds["group"], all_labels, openai_classnames, [compare_class], compare_class_A, model, device=device)

    #         print(f"=== Simple Method Evaluation for {compare_class} with '{compare_class_B}' label===")
    #         print("Feature Embedding:")
    #         evaluate_group_emb_similarity(extra_embeds["feature"], all_labels, openai_classnames, [compare_class], compare_class_B, model, device=device)
    #         print("Negative Embedding:")
    #         evaluate_group_emb_similarity(extra_embeds["negative"], all_labels, openai_classnames, [compare_class], compare_class_B, model, device=device)
    #         print("Group Embedding:")
    #         evaluate_group_emb_similarity(extra_embeds["group"], all_labels, openai_classnames, [compare_class], compare_class_B, model, device=device)
    # # method 에 proj 라는 단어가 있다면
    # elif 'proj' in args.method:
    #     for compare_class in compare_classes:
    #         print(f"=== Projection Method Evaluation for {compare_class} with '{compare_class_A}' label===")
    #         print("Raw (unadjusted) Embedding (top-k candidates):")
    #         evaluate_topk_emb_similarity(extra_embeds["raw"], all_labels, openai_classnames, [compare_class], compare_class_A, model, device=device)
    #         print("Orthogonal Embedding (top-k candidates):")
    #         evaluate_topk_emb_similarity(extra_embeds["orthogonal"], all_labels, openai_classnames, [compare_class], compare_class_A, model, device=device)
    #         print("Projection Embedding (top-k candidates):")
    #         evaluate_topk_emb_similarity(extra_embeds["projection"], all_labels, openai_classnames, [compare_class], compare_class_A, model, device=device)

    #         print(f"=== Projection Method Evaluation for {compare_class} with '{compare_class_B}' label===")
    #         print("Raw (unadjusted) Embedding (top-k candidates):")
    #         evaluate_topk_emb_similarity(extra_embeds["raw"], all_labels, openai_classnames, [compare_class], compare_class_B, model, device=device)
    #         print("Orthogonal Embedding (top-k candidates):")
    #         evaluate_topk_emb_similarity(extra_embeds["orthogonal"], all_labels, openai_classnames, [compare_class], compare_class_B, model, device=device)
    #         print("Projection Embedding (top-k candidates):")
    #         evaluate_topk_emb_similarity(extra_embeds["projection"], all_labels, openai_classnames, [compare_class], compare_class_B, model, device=device)
    # else:
    #     print("No extra evaluation for the given method.")