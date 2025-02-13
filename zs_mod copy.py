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

def collect_topk_predictions(dataloader, model, zeroshot_weights, classnames, k=5, device='cuda'):
    """
    각 ground-truth label(클래스)마다, 모델이 예측한 top-k 클래스들을 누적한 뒤,
    그 중 가장 많이 등장한 상위 k개를 모아 dictionary 형태로 반환.
    """
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

# Collect top-k predictions for each class (if not cached)
topk_pred_path = f'./results/{args.dataset}_top{args.topk}_predictions_{args.model.replace("/", "_").replace("-", "_")}.json'
if not os.path.exists(topk_pred_path):
    zeroshot_weights = zeroshot_classifier(args, model, openai_classnames, openai_imagenet_template)
    topk_predictions = collect_topk_predictions(
        dataloader, model, zeroshot_weights, openai_classnames,
        k=args.topk, device='cuda'
    )
    os.makedirs('./results', exist_ok=True)
    with open(topk_pred_path, 'w') as f:
        json.dump(topk_predictions, f, indent=4)
else:
    with open(topk_pred_path, 'r') as f:
        topk_predictions = json.load(f)

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

# ----------------------------------------------------------------
# 3) Create new "modified" text embeddings using top-5 info
# ----------------------------------------------------------------
def create_modified_embedding(
    model, main_class, top5_dict, 
    alpha=1.0, beta=1.0, svd_threshold=0.95, 
    device='cuda'
):

def create_modified_embedding_v1(
    model, main_class, top5_dict, 
    alpha=1.0, beta=1.0, svd_threshold=0.95, 
    device='cuda'
):
    """
    GradOrth 방식(Orthogonal Projection)을 활용하여
    main_class A의 텍스트 임베딩을 재조정.
    
    1) top-5 클래스 [A, B, C, D, E] 수집
    2) B,C,D,E 임베딩 행렬 R 생성 -> SVD
    3) A 임베딩을 R로부터 정의된 subspace에 투영/직교 분해
    4) (alpha, beta) 조합으로 A를 강화/감쇠 후 최종 벡터 반환
    """
    # 1) gather top-5 classes
    neighbors = top5_dict.get(main_class, [])
    if main_class not in neighbors:
        neighbors = [main_class] + neighbors
    
    # 중복 제거 & top-5 맞추기
    neighbors = list(dict.fromkeys(neighbors))  # preserves order
    if len(neighbors) < 5:
        neighbors += [main_class] * (5 - len(neighbors))
    else:
        neighbors = neighbors[:5]
    
    # ensure main_class is at position 0
    if neighbors[0] != main_class:
        if main_class in neighbors:
            neighbors.remove(main_class)
        neighbors.insert(0, main_class)
    
    A, B, C, D, E = neighbors
    
    # 2) get embeddings
    eA = get_text_embedding(model, [A], device=device)  # shape [dim,]
    eB = get_text_embedding(model, [B], device=device)
    eC = get_text_embedding(model, [C], device=device)
    eD = get_text_embedding(model, [D], device=device)
    eE = get_text_embedding(model, [E], device=device)
    
    # ----- Build R from B,C,D,E  -----
    # let's stack them as columns => R shape: [dim, 4]
    # (PyTorch에서는 svd를 R행렬이 (m,n)일 때 m >= n인 편이 낫습니다.)
    R = torch.stack([eB, eC, eD, eE], dim=1).float()  # shape: (dim, 4)
    
    # 3) SVD
    # torch.linalg.svd는 full_matrices=False로 두면 
    # U.shape == (dim,4), S.shape == (4,), V.shape == (4,4)
    U, S, Vt = torch.linalg.svd(R, full_matrices=False)  # R = U S V^T
    
    # ----- Determine rank k via threshold -----
    # Frobenius norm or singular values 합으로 임계 결정
    # sum(S[:k]^2) / sum(S^2) >= svd_threshold
    total_energy = torch.sum(S**2)
    cum_energy = torch.cumsum(S**2, dim=0)
    k = torch.searchsorted(cum_energy, svd_threshold * total_energy).item() + 1
    k = min(k, S.shape[0])  # 최대 4
    
    # 상위 k개만 쓰는 서브스페이스
    U_k = U[:, :k]  # shape (dim, k)
    
    # 4) Project eA onto subspace(U_k)
    #    proj(eA) = U_k (U_k^T eA)
    eA_proj = U_k @ (U_k.T @ eA.float())    # A의 "공통 성분"
    eA_orth = eA - eA_proj         # A의 "고유 성분"
    
    # (선택) B,C,D,E의 평균벡터
    eBCDE = (eB + eC + eD + eE) / 4.0
    
    # B,C,D,E 공통성 or 고유성도 쓰고 싶다면:
    eBCDE_proj = U_k @ (U_k.T @ eBCDE.float())  # B,C,D,E 공통
    eBCDE_orth = eBCDE - eBCDE_proj     # B,C,D,E 특이 성분
    
    # 5) 최종 결합
    #   eA_mod = eA_proj + alpha * eA_orth - beta * eBCDE_orth
    #   (아래는 예시이므로, 원하는 형태로 조합 가능)
    eA_mod = eA_proj + alpha * eA_orth - beta * eBCDE_orth
    # L2 정규화
    eA_mod = eA_mod / eA_mod.norm(dim=0, p=2)
    
    return eA_mod.half()

def create_modified_embedding_v2(
    model,
    main_class,
    top5_dict,
    alpha=1.2,        # factor to "enhance" the A-direction
    beta=1.2,         # factor to "weaken" the BCDE-direction
    device='cuda'
):
    """
    Completely overhauled version of create_modified_embedding:
      1) Collect embeddings for A,B,C,D,E.
      2) Compute SVD of all five embeddings stacked into a matrix.
      3) Compute SVD (or trivial direction) of A alone, and SVD of BCDE together.
      4) Identify which principal directions in step (2) align best
         with A's top direction and with BCDE's top direction,
         then rescale those singular values to emphasize A and de-emphasize BCDE.
      5) Reconstruct from modified SVD and return the updated version of A's embedding.
    """
    # -----------------------------
    # 1) gather top-5 classes
    neighbors = top5_dict.get(main_class, [])
    if main_class not in neighbors:
        neighbors = [main_class] + neighbors

    # deduplicate & ensure at least length 5
    neighbors = list(dict.fromkeys(neighbors))
    if len(neighbors) < 5:
        neighbors += [main_class] * (5 - len(neighbors))
    else:
        neighbors = neighbors[:5]

    # ensure main_class is at position 0
    if neighbors[0] != main_class:
        if main_class in neighbors:
            neighbors.remove(main_class)
        neighbors.insert(0, main_class)

    A, B, C, D, E = neighbors  # short-hand

    # -----------------------------
    # 2) Get embeddings for A,B,C,D,E
    eA = get_text_embedding(model, [A], device=device)  # shape [dim]
    eB = get_text_embedding(model, [B], device=device)
    eC = get_text_embedding(model, [C], device=device)
    eD = get_text_embedding(model, [D], device=device)
    eE = get_text_embedding(model, [E], device=device)

    # stack them into shape [5, dim], so each row is one embedding
    all_matrix = torch.stack([eA, eB, eC, eD, eE], dim=0).float()  # [5, dim]

    # -----------------------------
    # 2) SVD of all embeddings
    # all_matrix = U * diag(S) * V^T
    # where U is [5 x 5], S is [5], V is [dim x 5].
    U, S, Vt = torch.svd(all_matrix)  # PyTorch: returns U, S, V
    # in some PyTorch versions, it's torch.linalg.svd(all_matrix, full_matrices=False)
    #   => U: [5 x 5], S: [5], Vt: [dim x 5]  (note that older versions have V not V^T)

    # -----------------------------
    # 3) Directions from A alone & from BCDE
    # A alone is just a single vector shape [dim], so its "top singular vector" is
    # simply the normalized eA. Similarly, for BCDE, we get the top singular vector from SVD.

    vA = eA / eA.norm()  # direction of A
    bcde_matrix = torch.stack([eB, eC, eD, eE], dim=0).float()  # shape [4, dim]
    U_bcde, S_bcde, V_bcde = torch.svd(bcde_matrix)     # for BCDE
    # top singular vector => V_bcde[:, 0], shape [dim]
    vBCDE = V_bcde[:, 0]

    # -----------------------------
    # 4) Identify which principal directions in the full SVD best match vA and vBCDE.
    #
    # The matrix 'Vt' is size [dim x 5], so each column in Vt is one principal direction
    # in embedding space. We'll check dot products with vA, vBCDE to find the best match.
    #
    # If you used torch.linalg.svd, it might give V of shape [dim, 5],
    # so be mindful whether you need V or V^T. In older PyTorch, 'V' from torch.svd is
    # actually the matrix of right singular vectors in columns, i.e. shape [dim x 5].
    # The example here assumes older torch.svd usage: Vt is shape [dim x 5].

    V = Vt  # rename for clarity
    # dot shape: V is [dim x 5], vA is [dim], => V.T x vA is [5]
    dot_with_A    = torch.mv(V.t(), vA.float()).abs()     # [5]
    dot_with_BCDE = torch.mv(V.t(), vBCDE.float()).abs()  # [5]

    iA    = torch.argmax(dot_with_A)    # index in [0..4] that best matches A
    iBCDE = torch.argmax(dot_with_BCDE) # index in [0..4] that best matches BCDE direction

    # Copy S so we can adjust it
    S_prime = S.clone()

    # Strengthen the direction that aligns with A
    S_prime[iA]    = S_prime[iA]    * alpha

    # Weaken the direction that aligns with BCDE
    S_prime[iBCDE] = S_prime[iBCDE] / beta

    # -----------------------------
    # 5) Reconstruct with the adjusted singular values
    # E' = U * diag(S_prime) * V^T
    # shape of E' is again [5, dim], i.e. each row is an updated embedding
    S_diag = torch.diag(S_prime)               # [5 x 5]
    E_prime = U.mm(S_diag).mm(V.t())           # [5, dim]
    
    # Updated A is row 0
    eA_prime = E_prime[0]  # shape [dim]
    # Optionally L2-normalize
    eA_mod = eA_prime / eA_prime.norm(p=2)

    # Return the final re-scaled embedding for A
    # (Or you could return all 5 if you want them.)
    return eA_mod.half()

def create_modified_embedding_v3(
    model,
    main_class,
    top5_dict,
    alpha=1.2,        # factor to "enhance" the A-direction
    beta=1.2,         # factor to "weaken" the BCDE-direction
    device='cuda'
):
    """
    Modify the text embedding of a main class by enhancing its direction
    and weakening the directions of its top-5 neighbors.
    """

    # -----------------------------
    # 1) Gather top-5 classes
    neighbors = top5_dict.get(main_class, [])
    if main_class not in neighbors:
        neighbors = [main_class] + neighbors

    # Deduplicate & ensure at least length 5
    neighbors = list(dict.fromkeys(neighbors))
    if len(neighbors) < 5:
        neighbors += [main_class] * (5 - len(neighbors))
    else:
        neighbors = neighbors[:5]

    # Ensure main_class is at position 0
    if neighbors[0] != main_class:
        if main_class in neighbors:
            neighbors.remove(main_class)
        neighbors.insert(0, main_class)

    A, B, C, D, E = neighbors  # short-hand

    # -----------------------------
    # 2) Get embeddings for A, B, C, D, E
    eA = get_text_embedding(model, [A], device=device)  # shape [dim]
    eB = get_text_embedding(model, [B], device=device)
    eC = get_text_embedding(model, [C], device=device)
    eD = get_text_embedding(model, [D], device=device)
    eE = get_text_embedding(model, [E], device=device)

    # Stack them into shape [5, dim], so each row is one embedding
    all_matrix = torch.stack([eA, eB, eC, eD, eE], dim=0).float()  # [5, dim]

    # -----------------------------
    # 3) Orthonormalize the basis using Gram-Schmidt
    # -----------------------------
    def gram_schmidt(matrix):
        # Applies Gram-Schmidt process row-wise to orthonormalize the matrix
        basis = []
        for i in range(matrix.size(0)):
            vec = matrix[i]
            for b in basis:
                vec = vec - torch.dot(vec, b) * b  # Subtract projection
            vec = vec / vec.norm(p=2)  # Normalize
            basis.append(vec)
        return torch.stack(basis)

    normalized_matrix = gram_schmidt(all_matrix)  # [5, dim]

    # -----------------------------
    # 4) Modify Embedding for A
    # -----------------------------

    # Calculate the projection of eA onto each of the embeddings
    coefficients = torch.matmul(normalized_matrix, eA.float())  # shape [5]

    # Modify coefficients:
    mod_coefficients = coefficients.clone()
    mod_coefficients[0] = mod_coefficients[0] * alpha  # Enhance A's direction
    mod_coefficients[1:] = mod_coefficients[1:] * beta  # Weaken neighbors' directions
    # import pdb; pdb.set_trace()
    # Reconstruct the modified embedding eA'
    eA_prime = torch.matmul(mod_coefficients.float().unsqueeze(0), normalized_matrix.float()).squeeze(0)  # shape [dim]

    # -----------------------------
    # 6) Normalize the Modified Embedding
    # -----------------------------
    eA_mod = eA_prime / eA_prime.norm(p=2)  # L2-normalization
    import pdb; pdb.set_trace()
    return eA_mod.half()  # Return in half precision

def create_modified_embedding_v4(
    model,
    main_class,
    top5_dict,
    alpha=1.2,   # A 방향(혹은 공통 성분)을 얼마나 강화할지
    beta=0.8,    # negative 방향을 얼마나 가중할지
    device='cuda'
):
    """
    5개 텍스트 임베딩(A,B,C,D,E)에서:
      - PCA의 최상위 주성분을 공통성분으로 삼고
      - B,C,D,E 평균 - A 의 차이를 hard negative로 보고
      - A 임베딩을 적절히 조합해 최종 수정 임베딩(eA_mod)을 얻는다.
    """
    # -----------------------------
    # 1) Gather top-5 classes
    neighbors = top5_dict.get(main_class, [])
    # main_class가 neighbors에 없다면 맨 앞에 추가
    if main_class not in neighbors:
        neighbors = [main_class] + neighbors

    # 중복 제거 & 최소 5개 맞추기
    neighbors = list(dict.fromkeys(neighbors))
    if len(neighbors) < 5:
        neighbors += [main_class] * (5 - len(neighbors))
    else:
        neighbors = neighbors[:5]

    # 맨 앞은 반드시 main_class
    if neighbors[0] != main_class:
        if main_class in neighbors:
            neighbors.remove(main_class)
        neighbors.insert(0, main_class)

    A, B, C, D, E = neighbors  # 단순한 표기

    # -----------------------------
    # 2) Get embeddings for A, B, C, D, E
    eA = get_text_embedding(model, [A], device=device).float()  # shape [dim]
    eB = get_text_embedding(model, [B], device=device).float()
    eC = get_text_embedding(model, [C], device=device).float()
    eD = get_text_embedding(model, [D], device=device).float()
    eE = get_text_embedding(model, [E], device=device).float()

    # (dim) 형태 -> [5, dim]으로 쌓기
    all_matrix = torch.stack([eA, eB, eC, eD, eE], dim=0)  # [5, dim]

    # -----------------------------
    # 3) PCA를 통해 "공통 성분(Top PC)" 찾기
    # -----------------------------
    # (1) 평균 제거
    mean_vec = all_matrix.mean(dim=0)           # shape [dim]
    centered = all_matrix - mean_vec.unsqueeze(0)  # [5, dim], 각 임베딩에서 평균 뺀 것

    # (2) SVD 혹은 torch.pca_lowrank 등 사용
    #    여기서는 torch.svd 예시
    #    centered = U @ diag(S) @ V^T  (torch 1.9+에서는 torch.linalg.svd 권장)
    U, S, V = torch.svd(centered.float())   # V: [dim, dim], V[:,0]이 첫 번째 주성분 (PC1)

    # 첫 번째 주성분(PC1)을 공통 방향으로 본다
    top_pc = V[:, 0]                # shape [dim]
    top_pc = top_pc / top_pc.norm() # 정규화

    # A 벡터를 "평균 제거"한 뒤, 그 PC에 대한 투영 = 공통 성분
    eA_centered = eA - mean_vec
    dot_A = torch.dot(eA_centered, top_pc)      # 스칼라
    eCommon = dot_A * top_pc                    # A가 PC1 위로 투영된 벡터

    # -----------------------------
    # 4) "Hard negative" 추출
    # -----------------------------
    # (1) B,C,D,E 평균벡터
    eBCDE = (eB + eC + eD + eE) / 4.0
    eBCDE_centered = eBCDE - mean_vec

    # (2) 공통성분 방향은 제외하고, 나머지 방향으로 차이를 뽑는다
    #     즉 B,C,D,E 평균 - A에서 PC1 성분을 제거한 "orth 부분"을 hard negative로 본다
    proj_bcde = torch.dot(eBCDE_centered, top_pc) * top_pc
    bcde_orth = eBCDE_centered - proj_bcde      # BCDE의 공통성분 제거 후 (직교)
    
    # A의 직교 부분도 구한다 (A에서 공통성분 제거)
    a_orth = eA_centered - eCommon
    
    # bcde_orth - a_orth => "BCDE가 갖고 A는 갖지 않는" 방향의 잔차
    eHardNeg = bcde_orth - a_orth

    # -----------------------------
    # 5) 최종 A 임베딩 조합
    #    eA_centered + alpha * eCommon + beta * eHardNeg
    # -----------------------------
    eA_prime_centered = eA_centered + alpha * eCommon - beta * eHardNeg
    # 다시 평균 복원
    eA_prime = eA_prime_centered + mean_vec

    # -----------------------------
    # 6) Normalize the Modified Embedding
    # -----------------------------
    eA_mod = eA_prime / eA_prime.norm(p=2)  # L2-normalization
    
    return eA_mod.half()  # Return in half precision

def create_modified_embedding_v5(
    model, main_class, top5_dict, 
    alpha=1.0, beta=1.0, svd_threshold=0.95, 
    device='cuda'
):
    """
    수정된 버전: 공통 특징은 유지하면서 고유 특징을 강화.
    """

    # 1) Gather embeddings for main_class and top_classes
    neighbors = top5_dict.get(main_class, [])
    neighbors = [main_class] + neighbors
    neighbors = list(dict.fromkeys(neighbors))  # Remove duplicates, preserve order
    
    while len(neighbors) < 5:
        neighbors.append(main_class)

    # Split into A (main_class) and B, C, D, E
    A = neighbors[0]
    B, C, D, E = neighbors[1:5]

    # Extract embeddings for each class
    eA = get_text_embedding(model, [A], device=device)  # shape [dim,]
    eB = get_text_embedding(model, [B], device=device)
    eC = get_text_embedding(model, [C], device=device)
    eD = get_text_embedding(model, [D], device=device)
    eE = get_text_embedding(model, [E], device=device)

    # 2) Create R matrix and apply mean-centering
    R = torch.stack([eA, eB, eC, eD, eE], dim=1).float()  # shape (dim, 5)
    mean_vec = torch.mean(R, dim=1, keepdim=True)  # Compute mean vector
    R_centered = R - mean_vec  # Mean centering

    # 3) Perform SVD to find shared subspace (U_k)
    U, S, Vt = torch.linalg.svd(R_centered, full_matrices=False)  # R_centered = U S V^T
    total_energy = torch.sum(S**2)
    cum_energy = torch.cumsum(S**2, dim=0)
    k = torch.searchsorted(cum_energy, svd_threshold * total_energy).item() + 1
    k = min(k, S.shape[0])
    U_k = U[:, :k]  # Top-k singular vectors (shared subspace)

    # 4) Project embeddings onto the shared subspace
    eA_proj = U_k @ (U_k.T @ (eA - mean_vec.squeeze(1)))  # A's shared component
    eA_orth = eA - mean_vec.squeeze(1) - eA_proj          # A's unique component
    eBCDE = (eB + eC + eD + eE) / 4.0
    eBCDE_proj = U_k @ (U_k.T @ (eBCDE - mean_vec.squeeze(1)))  # Shared component of B, C, D, E

    # 5) Combine components to form modified embedding
    eA_proj = eA_proj / eA_proj.norm(p=2)
    eA_orth = eA_orth / eA_orth.norm(p=2)
    eBCDE_proj = eBCDE_proj / eBCDE_proj.norm(p=2)
    
    # Adjust final embedding
    eA_mod = (
        alpha * eA_orth +  # 강화된 고유 성분
        (1 - beta) * eA_proj -  # 공통 성분 보존
        beta * eBCDE_proj  # 나머지 성분 제거
    )

    # Normalize final embedding
    eA_mod = eA_mod / eA_mod.norm(p=2)
    
    return eA_mod.half()



# ----------------------------------------------------------------
# 4) Build a new zero-shot classifier from these modified embeddings
# ----------------------------------------------------------------
def build_modified_classifier(model, classnames, top5_dict, alpha=1.0, beta=1.0, device='cuda'):
    """
    For each of the 1000 ImageNet classnames, create a modified embedding,
    then stack them together -> shape (dim, #classes).
    """
    print("[INFO] Building modified embeddings ...")
    new_weights = []
    for cname in tqdm(classnames):
        e_mod = create_modified_embedding_v5(
            model, 
            main_class=cname, 
            top5_dict=top5_dict,
            alpha=alpha,
            beta=beta,
            device=device
        )
        new_weights.append(e_mod)
    # stack along dim=1 => (D, #classes)
    new_weights = torch.stack(new_weights, dim=1).to(device)
    return new_weights

modified_weights = build_modified_classifier(
    model, 
    openai_classnames, 
    top5_dict=topk_predictions,
    alpha=args.alpha, 
    beta=args.beta,
    device='cuda'
)

print(f"[INFO] Modified zeroshot weights shape: {modified_weights.shape}")

# ----------------------------------------------------------------
# 5) Evaluate the new embeddings and collect top-5 predictions
# ----------------------------------------------------------------
top1_mod, top5_mod, n_mod = 0., 0., 0.
top5_modified_predictions = {classname: [] for classname in openai_classnames}  # To store top-5 predictions
model.eval()

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Evaluating modified embeddings"):
        batch = maybe_dictionarize_batch(batch)
        images, labels = batch['images'].cuda(), batch['labels'].cuda()

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Use modified weights
        logits = 100. * image_features @ modified_weights

        # Compute accuracies
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        top1_mod += acc1
        top5_mod += acc5
        n_mod += images.size(0)

        # Collect top-5 predictions for each image
        _, top5_indices = logits.topk(5, dim=1, largest=True, sorted=True)
        for label, top5_idx in zip(labels, top5_indices):
            true_classname = openai_classnames[label.item()]
            predicted_classnames = [openai_classnames[idx.item()] for idx in top5_idx]
            top5_modified_predictions[true_classname].extend(predicted_classnames)

# Compute final accuracy percentages
top1_mod = (top1_mod / n_mod) * 100
top5_mod = (top5_mod / n_mod) * 100

# Save top-5 predictions to JSON
modified_topk_pred_path = f'./results/Modified_{args.dataset}_top{args.topk}_predictions_{args.model.replace("/", "_").replace("-", "_")}_alpha_{args.alpha}_beta_{args.beta}.json'
os.makedirs('./results', exist_ok=True)

# Keep only the top-5 most common predictions for each class
top5_most_common_mod = {
    classname: [item for item, _ in Counter(predictions).most_common(5)]
    for classname, predictions in top5_modified_predictions.items()
}

with open(modified_topk_pred_path, 'w') as f:
    json.dump(top5_most_common_mod, f, indent=4)

print(f"[INFO] Modified Top-5 predictions saved to {modified_topk_pred_path}")
print(f"[MODIFIED EMBEDDINGS] Top-1 accuracy: {top1_mod:.2f}")
print(f"[MODIFIED EMBEDDINGS] Top-5 accuracy: {top5_mod:.2f}")

# ----------------------------------------------------------------
# 6) (Optional) Log results with wandb
# ----------------------------------------------------------------
if not args.ignore_wandb:
    wandb.init(entity="mlai_medical_ai", project="COE", config=args)
    wandb.config.update(args)
    run_name = f"{args.model.replace('/', '_').replace('-', '_')}_alpha_{args.alpha}_beta_{args.beta}"
    wandb.run.name = run_name
    wandb.log({
        "Modified Top-1 accuracy": top1_mod,
        "Modified Top-5 accuracy": top5_mod
    })

