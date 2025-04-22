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
        default=os.path.expanduser('/data1/bubble3jh/data/'),
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
        "--custom_template", default='default', type=str,
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






# ----------------------------------------------------------------
# 5) Evaluate the new embeddings and collect top-5 predictions
# ----------------------------------------------------------------


def main():
    args = parse_arguments()
    device = 'cuda'
    
    if args.custom_template == 'default':    
        template = openai_imagenet_template
    elif args.custom_template == 'simple':
        template = [lambda x : f"a photo of a {x}."]
    elif args.custom_template == 'class':
        template = [lambda x : f"{x}."]
    else:
        raise ValueError(f"Unknown template: {args.custom_template}")
    model, preprocess = clip.load(args.model, device=device)

    print(f"{len(openai_classnames)} classes, {len(template)} templates")

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
        zeroshot_weights = zeroshot_classifier(args, model, openai_classnames, template)
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
    

    zeroshot_weights = zeroshot_classifier(args, model, openai_classnames, template)
    class_embeddings = zeroshot_weights.t().cuda()  # (C, D)

    for alpha in [1.0,]:
        for beta in [0.8, 0]:
            for method in ['text_svd_proj']:
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