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
from jh_utils import *
import wandb
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-location", type=str, default=os.path.expanduser('/data1/bubble3jh/data/'))
    parser.add_argument("--model-location", type=str, default=os.path.expanduser('./checkpoints'))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--method", type=str, default='svd_proj')
    parser.add_argument("--dataset",  default="ImageNet")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default='ViT-B/32')
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--custom_template", default="openai")
    parser.add_argument("--use_wandb", action='store_true', default=False)
    parser.add_argument("--mean_center", action='store_false', default=True)
    parser.add_argument("--pc_mode", type=str, default='upper')
    parser.add_argument("--top_l", type=int, default=-1)
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = 'cuda'
    if args.custom_template == "openai":
        template = openai_imagenet_template
    elif args.custom_template == 'simple':
        template = [lambda x : f"a photo of a {x}."]
    elif args.custom_template == 'class':
        template = [lambda x : f"{x}."]
    model, preprocess = clip.load(args.model, device=device)
    print(f"{len(openai_classnames)} classes, {len(template)} templates")
    print("[INFO] Loaded model")
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
    zeroshot_weights = load_text_checkpoint(args, model, template, device=device)
    
    # Image-wise top-k 인덱스 수집
    topk_indices = load_topk_predictions(args, model, dataloader, zeroshot_weights, device=device)

    # 이미지 임베딩 및 레이블 계산
    img_embs, all_labels = load_image_embeddings(args, model, dataloader, device=device)

    # class_embeddings: (C, D)
    class_embeddings = zeroshot_weights.t().to(device)

    # 수정된 텍스트 임베딩 및 extra embedding 생성
    modified_embs, mean_cos_sim, extra_embeds = build_modified_text_embs_vectorized(
        args,
        class_embeddings=class_embeddings,
        topk_indices=topk_indices,
        alpha=args.alpha,
        beta=args.beta,
        method=args.method,
        device=device,
        img_embs=img_embs,
        batch_size=args.batch_size,
        null_text_emb=get_null_text_embedding(model, template, device) if 'unknown' in args.method else None
    )
    print(f"[INFO] Alpha: {args.alpha}, Beta: {args.beta}, Method: {args.method}")
    print(f"[INFO] Modified Cosine Similarity: {mean_cos_sim}")

    
    # 예시로 main() 함수 내에 projection 계열(method가 'qr_proj', 'svd_proj', 'svd_mm_proj', 'geodesic_svd_mm_proj')일 때
    # extra_embeds 딕셔너리에 "raw"와 "projection" 키가 존재하므로 아래와 같이 호출할 수 있습니다.
    if 'text' in args.method:
        print("=== Raw vs. Projection Cosine Similarity Evaluation ===")
        avg_cos_sim, var_cos_sim = evaluate_raw_projection_cosine_similarity(extra_embeds["raw"], extra_embeds["projection"])

    original_topk_embs = extra_embeds["raw"]  # (N, k, D)
    modified_topk_embs = modified_embs  # (N, k, D)
    
    mean_pairwise_orig, mean_pairwise_mod, mean_candidate_preservation = validate_adjustment_effectiveness(original_topk_embs, modified_topk_embs)
    
    top1_mod, top5_mod = evaluate_modified_embeddings_vectorized(
        args,
        dataloader,
        model,
        modified_topk_embs,
        topk_indices,
        class_embeddings=zeroshot_weights,
        device='cuda',
        img_embs=img_embs,
        all_labels=all_labels
    )
    
    if args.use_wandb:
        wandb.init(entity="mlai_medical_ai", project="COE", config=args, reinit=True)
        wandb.config.update(args)
        run_name = f"{args.model.replace('/', '_').replace('-', '_')}_{args.method}_alpha_{args.alpha}_beta_{args.beta}"
        wandb.run.name = run_name
        wandb.log({
            "Modified Top-1 accuracy": top1_mod,
            "Modified Top-5 accuracy": top5_mod,
            "Modified Cosine Similarity": mean_cos_sim,
            "Raw vs. Projection Cosine Similarity": avg_cos_sim,
            "Raw vs. Projection Cosine Similarity Variance": var_cos_sim,
            "Pairwise Original Top-k Embedding Preservation": mean_pairwise_orig,
            "Pairwise Modified Top-k Embedding Preservation": mean_pairwise_mod,
            "Candidate Preservation": mean_candidate_preservation
        })
if __name__ == "__main__":
    main()
