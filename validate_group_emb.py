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

def build_modified_text_embs_vectorized(
    class_embeddings,  # (C, D)
    topk_indices,      # (N, k)
    alpha=1.0,
    beta=1.0,
    method='simple',
    device='cuda',
    img_embs=None,
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

    modified_embs_list = []   # (B, k, D)
    group_embs_list = []      # (B, D)
    cos_sim_list = []         # scalar per batch

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

        else:
            raise ValueError("Unknown method")

        if method != 'geodesic_svd_mm_proj':
            cos_sim = F.cosine_similarity(modified_embs, topk_embs, dim=-1).mean()
        modified_embs_list.append(modified_embs.cpu())
        group_embs_list.append(group_embs.cpu())
        cos_sim_list.append(cos_sim.item())

    modified_embs_full = torch.cat(modified_embs_list, dim=0).to(device)
    group_embs_full = torch.cat(group_embs_list, dim=0).to(device)
    mean_cos_sim = sum(cos_sim_list) / len(cos_sim_list)

    # extra_embeds dict 생성
    if method == 'simple':
        extra_embeds = {
            "feature": torch.cat(extra_feature_list, dim=0).to(device),
            "negative": torch.cat(extra_negative_list, dim=0).to(device),
            "group": group_embs_full  # 이미 computed above
        }
    # proj 라는 단어가 있다면
    elif 'proj' in method:
        extra_embeds = {
            "raw": torch.cat(extra_raw_list, dim=0).to(device),  # 조정 전 top-k embedding
            "orthogonal": torch.cat(extra_orthogonal_list, dim=0).to(device),
            "projection": torch.cat(extra_projection_list, dim=0).to(device)
        }
    else:
        extra_embeds = None

    return modified_embs_full, mean_cos_sim, extra_embeds

#########################################
# 2. 텍스트 임베딩 생성 함수 (이미 제공됨)
#########################################

def get_text_embedding(model, cls_list, device='cuda'):
    """
    cls_list 내의 클래스 이름에 대해 여러 템플릿(prompt)을 사용해 텍스트 임베딩의 평균값을 반환
    """
    texts = []
    for cname in cls_list:
        prompts = [t(cname) for t in openai_imagenet_template]
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

def evaluate_topk_emb_similarity(embeddings, all_labels, classnames, target_group, custom_text, model, device='cuda'):
    """
    embeddings: (N, k, D) – 각 이미지의 top-k 후보 embedding
    ground-truth 클래스가 target_group에 해당하는 이미지들에 대해,
    각 이미지의 k개의 후보와 get_text_embedding()으로 얻은 custom_text (예:"dog") 텍스트 임베딩과의 코사인 유사도를 계산.
    
    최종적으로, 각 이미지의 k개 유사도를 모두 모아서 평균 유사도를 출력합니다.
    """
    selected_sims = []
    text_emb = get_text_embedding(model, [custom_text], device=device)  # (D,)
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

#########################################
# 5. 예제: main() 내에서 사용하기
#########################################

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-location", type=str, default=os.path.expanduser('/mlainas/bubble3jh/data/'))
    parser.add_argument("--model-location", type=str, default=os.path.expanduser('./checkpoints'))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--method", type=str, default='simple')
    parser.add_argument("--dataset",  default="ImageNet")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default='ViT-B/32')
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()

def evaluate_raw_projection_cosine_similarity(raw_embeds, projection_embeds):
    """
    raw_embeds: (N, k, D) – 조정 전 top-k embedding
    projection_embeds: (N, k, D) – projection 벡터
    두 텐서의 각 대응되는 후보에 대해 코사인 유사도를 계산하고, 전체 평균 유사도를 출력합니다.
    """
    # 각 후보별로 코사인 유사도 계산 (마지막 차원 D 기준)
    cos_sims = F.cosine_similarity(raw_embeds, projection_embeds, dim=-1)  # 결과 shape: (N, k)
    avg_cos_sim = cos_sims.mean().item()
    print(f"Average cosine similarity between raw and projection vectors: {avg_cos_sim:.4f}")
    return avg_cos_sim


def main():
    args = parse_arguments()
    device = 'cuda'
    model, preprocess = clip.load(args.model, device=device)
    print(f"{len(openai_classnames)} classes, {len(openai_imagenet_template)} templates")
    print("[INFO] Loaded model")
    import pdb; pdb.set_trace()
    # 데이터셋 로딩
    dataset = getattr(datasets, args.dataset)(
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    dataloader = dataset.test_loader
    print("[INFO] Loaded dataset")
    # zero-shot 텍스트 임베딩 가중치 로딩/계산
    zeroshot_weights = None
    checkpoint_path = os.path.join(args.model_location, f"{args.dataset}_{args.model.replace('/', '_').replace('-', '_')}_text_embs.pt")
    if os.path.exists(checkpoint_path):
        print("[INFO] Loading zero-shot classifier from:", checkpoint_path)
        zeroshot_weights = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
    else:
        print("[INFO] Creating zero-shot classifier")
        zs_weights = []
        with torch.no_grad():
            for classname in openai_classnames:
                texts = [t(classname) for t in openai_imagenet_template]
                tokenized = clip.tokenize(texts).to(device)
                emb = model.encode_text(tokenized)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                zs_weights.append(emb.mean(dim=0) / emb.mean(dim=0).norm())
            zeroshot_weights = torch.stack(zs_weights, dim=1).to(device)
        torch.save(zeroshot_weights, checkpoint_path)
    
    # Image-wise top-k 인덱스 수집
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

    # 이미지 임베딩 및 레이블 계산
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

    # class_embeddings: (C, D)
    class_embeddings = zeroshot_weights.t().to(device)

    # 수정된 텍스트 임베딩 및 extra embedding 생성
    modified_embs, mean_cos_sim, extra_embeds = build_modified_text_embs_vectorized(
        class_embeddings=class_embeddings,
        topk_indices=topk_indices,
        alpha=args.alpha,
        beta=args.beta,
        method=args.method,
        device=device,
        img_embs=img_embs,
        batch_size=args.batch_size
    )
    print(f"[INFO] Alpha: {args.alpha}, Beta: {args.beta}, Method: {args.method}")
    print(f"[INFO] Modified Cosine Similarity: {mean_cos_sim}")

    
    # 예시로 main() 함수 내에 projection 계열(method가 'qr_proj', 'svd_proj', 'svd_mm_proj', 'geodesic_svd_mm_proj')일 때
    # extra_embeds 딕셔너리에 "raw"와 "projection" 키가 존재하므로 아래와 같이 호출할 수 있습니다.
    if args.method in ['qr_proj', 'svd_proj', 'svd_mm_proj', 'geodesic_svd_mm_proj']:
        print("=== Raw vs. Projection Cosine Similarity Evaluation ===")
        evaluate_raw_projection_cosine_similarity(extra_embeds["raw"], extra_embeds["projection"])


    # 평가: ground-truth가 dog 관련 클래스(예: 'Labrador Retriever', 'Golden Retriever', 'Maltese')인 경우
    # dog_group = ["Labrador Retriever", "Golden Retriever", "Maltese"]
    # custom_text = "truck"

    compare_classes = ["Granny Smith apple", "fig", "cucumber"]
    compare_class_A = "Green fruit"
    compare_class_B = "Red fruit"

    if args.method == 'simple':
        for compare_class in compare_classes:
            print(f"=== Simple Method Evaluation for {compare_class} with '{compare_class_A}' label===")
            print("Feature Embedding:")
            evaluate_group_emb_similarity(extra_embeds["feature"], all_labels, openai_classnames, [compare_class], compare_class_A, model, device=device)
            print("Negative Embedding:")
            evaluate_group_emb_similarity(extra_embeds["negative"], all_labels, openai_classnames, [compare_class], compare_class_A, model, device=device)
            print("Group Embedding:")
            evaluate_group_emb_similarity(extra_embeds["group"], all_labels, openai_classnames, [compare_class], compare_class_A, model, device=device)

            print(f"=== Simple Method Evaluation for {compare_class} with '{compare_class_B}' label===")
            print("Feature Embedding:")
            evaluate_group_emb_similarity(extra_embeds["feature"], all_labels, openai_classnames, [compare_class], compare_class_B, model, device=device)
            print("Negative Embedding:")
            evaluate_group_emb_similarity(extra_embeds["negative"], all_labels, openai_classnames, [compare_class], compare_class_B, model, device=device)
            print("Group Embedding:")
            evaluate_group_emb_similarity(extra_embeds["group"], all_labels, openai_classnames, [compare_class], compare_class_B, model, device=device)
    # method 에 proj 라는 단어가 있다면
    elif 'proj' in args.method:
        for compare_class in compare_classes:
            print(f"=== Projection Method Evaluation for {compare_class} with '{compare_class_A}' label===")
            print("Raw (unadjusted) Embedding (top-k candidates):")
            evaluate_topk_emb_similarity(extra_embeds["raw"], all_labels, openai_classnames, [compare_class], compare_class_A, model, device=device)
            print("Orthogonal Embedding (top-k candidates):")
            evaluate_topk_emb_similarity(extra_embeds["orthogonal"], all_labels, openai_classnames, [compare_class], compare_class_A, model, device=device)
            print("Projection Embedding (top-k candidates):")
            evaluate_topk_emb_similarity(extra_embeds["projection"], all_labels, openai_classnames, [compare_class], compare_class_A, model, device=device)

            print(f"=== Projection Method Evaluation for {compare_class} with '{compare_class_B}' label===")
            print("Raw (unadjusted) Embedding (top-k candidates):")
            evaluate_topk_emb_similarity(extra_embeds["raw"], all_labels, openai_classnames, [compare_class], compare_class_B, model, device=device)
            print("Orthogonal Embedding (top-k candidates):")
            evaluate_topk_emb_similarity(extra_embeds["orthogonal"], all_labels, openai_classnames, [compare_class], compare_class_B, model, device=device)
            print("Projection Embedding (top-k candidates):")
            evaluate_topk_emb_similarity(extra_embeds["projection"], all_labels, openai_classnames, [compare_class], compare_class_B, model, device=device)
    else:
        print("No extra evaluation for the given method.")

if __name__ == "__main__":
    main()
