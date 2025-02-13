import argparse
import os
import time
import torch
import clip
from tqdm import tqdm
import torch.nn.functional as F

import datasets
from utils import ModelWrapper, test_model_on_dataset, maybe_dictionarize_batch
from openai_imagenet_template import openai_imagenet_template


############################################
# 0. Projection helper functions
############################################

def project_naive(fvec, avec, alpha=1.0):
    """
    fvec: (D,) - 원본 임베딩
    avec: (D,) - ambiguous vector
    alpha: float
    """
    Ahat = avec / (avec.norm() + 1e-7)
    dotval = torch.dot(fvec, Ahat)
    proj = dotval * Ahat
    orth = fvec - proj
    # alpha=1 -> fvec 그대로
    fvec_prime = proj + alpha * orth
    return fvec_prime

def project_geodesic(fvec, avec, alpha=1.0):
    """
    fvec, avec: L2 정규화된 상태라 가정(혹은 내부에서 정규화)
    alpha=1 -> fvec, alpha=0 -> avec
    """
    f_norm = fvec.norm() + 1e-7
    a_norm = avec.norm() + 1e-7
    Fhat = fvec / f_norm
    Ahat = avec / a_norm
    cos_theta = torch.clamp(torch.dot(Fhat, Ahat), -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    if theta < 1e-6:
        return fvec  # 거의 같은 방향이면 그대로
    sin_theta = torch.sin(theta) + 1e-7
    f_factor = torch.sin((alpha) * theta) / sin_theta
    a_factor = torch.sin(1 - alpha * theta) / sin_theta
    fvec_prime = f_factor * Fhat + a_factor * Ahat
    return fvec_prime


############################################
# 1. Argument parser
############################################

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

    return parser.parse_args()


############################################
# 2. Zero-shot classifier builder 
############################################

def zeroshot_classifier(args, model, classnames, templates, device):
    checkpoint_path = f'./checkpoints/{args.dataset}_zs_cls_{args.model.replace("/", "_").replace("-", "_")}.pt'
    
    if os.path.exists(checkpoint_path):
        print('Loading zero-shot classifier.')
        zeroshot_weights = torch.load(checkpoint_path, weights_only=True)
    else:
        print('Building zero-shot classifier.')
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template(classname) for template in templates]
                tokens = clip.tokenize(texts).to(device)
                class_embeddings = model.encode_text(tokens)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        
        # Save the computed weights
        torch.save(zeroshot_weights, checkpoint_path)
    
    # shape: (D, num_classes) -> CLIP often uses (num_classes, D) but here we do t()
    return 100 * zeroshot_weights.t()


############################################
# 3. Ambiguous Vector Test function
############################################

def test_model_on_dataset_ambig(
    model, 
    dataset,
    zeroshot_weights,            # (num_classes, D)
    alpha=1.0,
    projection='naive',          # or 'geodesic'
    half_prec=0,
    adjust_target='text'         # "text" or "image"
):
    """
    기존 test_model_on_dataset와 유사하되:
    1) 첫 pass에서 top-5 예측
    2) 각 샘플별로 ambiguous vector A 계산 (top-5 클래스들의 평균 등)
    3) 'adjust_target'에 따라 텍스트 임베딩 or 이미지 임베딩을 투영
    4) 재계산한 logits으로 최종 accuracy 산출
    """
    model.eval()
    device = 'cuda'
    loader = dataset.test_loader

    if half_prec:
        zeroshot_weights = zeroshot_weights.half()  # 텍스트 임베딩 half
    else:
        zeroshot_weights = zeroshot_weights.float()
    
    num_classes, dim = zeroshot_weights.shape

    correct1, correct5, n = 0, 0, 0
    start_time = time.time()

    for i, batch in enumerate(tqdm(loader)):
        batch = maybe_dictionarize_batch(batch)
        inputs, labels = batch['images'].to(device), batch['labels'].to(device)
        if half_prec:
            inputs = inputs.half()

        # -----------------------------
        # 1) 1차 예측 (naive pass)
        # -----------------------------
        with torch.no_grad():
            logits_1st, features = model(inputs, return_features=True)
            # "features" = (batch_size, D) image embedding (assuming ModelWrapper is normalizing or not)
            top5_1st = logits_1st.topk(5, dim=1).indices  # (batch_size, 5)

        # -----------------------------
        # 2) Adjusting embeddings
        # -----------------------------
        # (batch_size, D)
        image_emb = torch.nn.functional.normalize(features, dim=1)  # 만약 필요시 normalize

        # 최종 로짓을 저장할 tensor
        # shape = (batch_size, num_classes)
        adjusted_logits = torch.zeros(inputs.size(0), num_classes, device=device)

        # 바깥 클래스(혹은 바깥 이미지) 로짓도 우선 naive pass 값으로 초기화
        # => (batch_size, num_classes)
        naive_logits = image_emb @ zeroshot_weights.transpose(0,1)
        adjusted_logits.copy_(naive_logits)

        # -----------------------------
        # 3) 샘플별 ambiguous projection
        # -----------------------------
        # top-5 안에 포함된 클래스들끼리 "ambiguous vector"를 만든 뒤
        # adjust_target에 따라, (A) 텍스트 임베딩을 조정하거나 (B) 이미지 임베딩을 조정
        # → 조정된 임베딩으로 로짓 갱신
        # -----------------------------
        for bidx in range(inputs.size(0)):
            c_top5 = top5_1st[bidx].tolist()  # shape(5,)

            # 해당 이미지 임베딩
            f_b = image_emb[bidx]  # shape(D,)

            # [Case 1] 텍스트 임베딩을 조정
            if adjust_target == 'text':
                # top-5 내 각 클래스 c_idx에 대해
                for c_idx in c_top5:
                    # ambiguous vector A = 평균(나머지 top5 클래스 임베딩)
                    others = [x for x in c_top5 if x != c_idx]
                    if len(others) == 0:
                        A = zeroshot_weights[c_idx].clone()
                    else:
                        A = zeroshot_weights[others, :].mean(dim=0)
                        A = A / (A.norm() + 1e-7)

                    # 원본 텍스트 임베딩 F
                    F = zeroshot_weights[c_idx]

                    # 투영
                    if projection == 'naive':
                        Fprime = project_naive(F, A, alpha=alpha).float()
                    else:
                        # geodesic
                        Fprime = project_geodesic(F, A, alpha=alpha).float()    

                    # image dot Fprime
                    new_logit = torch.dot(f_b, Fprime)
                    adjusted_logits[bidx, c_idx] = new_logit

            # [Case 2] 이미지 임베딩을 조정
            else:
                # ambiguous vector A = 평균(해당 이미지의 top-5 텍스트 임베딩)
                # => top-5 클래스들의 텍스트 임베딩 합/평균
                # => 그걸로 이미지 f_b를 투영
                # => 단, "클래스별로" 투영이 다르게끔 할 수도 있음(원본 질문에서 '말티즈를 추론할 땐 리트리버, 진돗개 평균' 이라 했으니),
                #    하지만 "이미지 1장에 top-5 클래스가 정해져 있으면, 하나의 A로 이미지 임베딩을 5가지 방식으로 각각 조정"?
                #    → 아래는 예시로, top-5 나머지 4개를 평균해서 "C에 대한" A를 구함
                #    (즉, text와 같은 방식)
                # => top-5 안 각 C에 대해 다른 f'를 써야 하므로, c_top5 루프 필요

                for c_idx in c_top5:
                    others = [x for x in c_top5 if x != c_idx]
                    if len(others) == 0:
                        A = zeroshot_weights[c_idx].clone()
                    else:
                        A = zeroshot_weights[others, :].mean(dim=0)
                        A = A / (A.norm() + 1e-7)
                    
                    # 투영 전, f_b는 (D,)
                    if projection == 'naive':
                        fprime = project_naive(f_b, A, alpha=alpha)
                    else:
                        fprime = project_geodesic(f_b, A, alpha=alpha)

                    # 그럼 fprime dot w_c
                    F_c = zeroshot_weights[c_idx]  # (D,)
                    new_logit = torch.dot(fprime, F_c)
                    adjusted_logits[bidx, c_idx] = new_logit

        # -----------------------------
        # 4) 최종 accuracy 계산
        # -----------------------------
        projection_fn = getattr(dataset, 'project_logits', None)
        if projection_fn is not None:
            adjusted_logits = projection_fn(adjusted_logits, device)

        if hasattr(dataset, 'project_labels'):
            labels_ = dataset.project_labels(labels, device)
        else:
            labels_ = labels

        pred_top1 = adjusted_logits.argmax(dim=1, keepdim=True)
        pred_top5 = adjusted_logits.topk(5, dim=1).indices

        correct_current1 = pred_top1.eq(labels_.view_as(pred_top1)).sum().item()
        correct1 += correct_current1

        correct_current5 = sum(labels_[jj] in pred_top5[jj] for jj in range(len(labels_)))
        correct5 += correct_current5

        n += labels_.size(0)

        if i % 20 == 0:
            percent_complete = 100.0 * i / len(loader)
            print(
                f"[{percent_complete:.0f}% {i}/{len(loader)}]\t"
                f"Top-1 Acc: {100 * (correct1/n):.2f}\t"
                f"Top-5 Acc: {100 * (correct5/n):.2f}"
            )

    top1 = correct1 / n
    top5 = correct5 / n
    elapsed = time.time() - start_time
    print(f"[DONE] Final Top-1: {100*top1:.2f}%, Top-5: {100*top5:.2f}%, Time: {elapsed:.1f}s")
    return top1, top5


############################################
# 4. Main
############################################

if __name__ == '__main__':
    args = parse_arguments()
    device = 'cuda'
    assert args.dataset in ['ImageNet']

    if args.custom_template:
        template = [lambda x : f"a photo of a {x}."]
    else:
        template = openai_imagenet_template

    # 1) Load base model
    base_model, preprocess = clip.load(args.model, device, jit=False)

    # 2) Load dataset
    dset = getattr(datasets, args.dataset)(
        preprocess, 
        location=args.data_location, 
        batch_size=args.batch_size, 
        num_workers=args.workers
    )

    # 3) Build zero-shot classifier weights
    clf = zeroshot_classifier(base_model, dset.classnames, template, device)  
    # clf shape: (num_classes, D)

    NUM_CLASSES = len(dset.classnames)
    feature_dim = base_model.visual.output_dim

    # 4) Wrap model so it returns (logits, features)
    #    By default, ModelWrapper does "image_emb dot clf" as logits
    model = ModelWrapper(base_model, feature_dim, NUM_CLASSES, normalize=True, initial_weights=clf)
    model = model.float().cuda()
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # 5) Run ambiguous projection test
    top1, top5 = test_model_on_dataset_ambig(
        model, 
        dset, 
        zeroshot_weights=clf, 
        alpha=args.alpha, 
        projection=args.orthogonal_projection, 
        half_prec=args.half_prec,
        adjust_target=args.adjust_target  # <-- "text" or "image"
    )

    print(f'[Final] Top1={100*top1:.2f}% Top5={100*top5:.2f}%')
