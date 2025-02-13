import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
import os
import pickle
from tqdm import tqdm
import clip
from openai_imagenet_template import openai_imagenet_template as template
from torchmetrics.classification import MulticlassCalibrationError
import copy
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pdb
import copy
import datasets
from clip.model import convert_weights


    
def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifier.cpu(), f)


def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None, rep_mask=None, rep_scale=None, clip_flag=True, lp_flag=False):
        super(ModelWrapper, self).__init__()
        self.model = model # model: CLIP class
        self.clip_flag = clip_flag
        if clip_flag:
            self.classification_head = torch.nn.Linear(feature_dim, num_classes)
            self.normalize = normalize
            if initial_weights is None:
                initial_weights = torch.zeros_like(self.classification_head.weight)
                torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
            self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
            self.classification_head.bias = torch.nn.Parameter(
                torch.zeros_like(self.classification_head.bias))
            self.rep_mask = rep_mask
            self.rep_scale = rep_scale
            if lp_flag:
                for name, param in self.model.named_parameters():
                    param.requires_grad_(False)
        else:
            self.feature_dim=feature_dim
            self.num_classes=num_classes
            if lp_flag:
                for name, param in self.model.named_parameters():
                    param.requires_grad_(False)
                
                self.model.head.weight.requires_grad_(True)
                self.model.head.bias.requires_grad_(True)

        #! discriminator for CLIP and other backbone
        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images, return_features=False):
        if self.clip_flag:
            # if self.rep_mask is not None:
            #     features = self.model.encode_image(images, self.rep_mask,  self.rep_scale)
            # else:
            features = self.model.encode_image(images)

            if self.normalize:
                features = features / features.norm(dim=-1, keepdim=True)
            logits = self.classification_head(features)
            if return_features:
                return logits, features
            return logits
        else:
            if return_features:
                features_ = self.model.forward_features(images)
                features = features_[:,0,:]
                features = self.model.fc_norm(features)
                features = self.model.head_drop(features)
                logits = self.model.head(features)
                ## pdb.set_trace()
                return logits, features
            return self.model(images)
            
    
    def save(self, filename):
        print(f'Saving classifier to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classifier from {filename}')
        return torch_load(filename)

def get_model_from_sd(state_dict, base_model, rep_mask=None, rep_scale=None, clip_flag=True):
    if clip_flag:
        feature_dim = state_dict['classification_head.weight'].shape[1]
        num_classes = state_dict['classification_head.weight'].shape[0]
    else:
        feature_dim = 768
        num_classes = 1000
    model = ModelWrapper(copy.deepcopy(base_model), feature_dim, num_classes, normalize=True, rep_mask=rep_mask, rep_scale=rep_scale, clip_flag=clip_flag)
    for p in model.parameters():
        p.data = p.data.float()
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()
    #devices = [x for x in range(torch.cuda.device_count())]
    return model #torch.nn.DataParallel(model,  device_ids=devices)

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None, shape=[512, 1000]):
        if weights is not None:
            output_size, input_size = weights.shape
            super().__init__(input_size, output_size)
        else:
            super().__init__(shape[0], shape[1])
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())

        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename, logger=None):
        print(f'Loading classification head from {filename}')
        if logger != None:
            logger.info(f'Loading classification head from {filename}')
        return torch_load(filename)

class ImageClassifier(torch.nn.Module):
    def __init__(self,
                 image_encoder,
                 classification_head,
                 process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs, out_feat=False):
        if self.process_images:
            feats = self.image_encoder(inputs)
        outputs = self.classification_head(feats)
        if out_feat:
            return outputs, feats
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return torch_load(filename)

def log_map_batch(base, x, eps=1e-7):
    """
    base: (B, D) or (B, 1, D) 형태  -- 여기선 (B,1,D)로 사용 가능
    x   : (B, M, D)               -- (M= k-1 or k+1 등)
    
    같은 batch 내에서, 각 sample별로 base_i, x_i에 대하여
    log_{base_i}(x_i)를 벡터화로 계산.
    
    반환: (B, M, D)
    """
    # 내적 & 각도
    # base: (B, 1, D), x: (B, M, D)
    dot_val = (base * x).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)  # (B, M, 1)
    theta = torch.acos(dot_val)                                     # (B, M, 1)
    
    sin_theta = torch.sin(theta)
    mask = (sin_theta < eps)  # (B, M, 1)
    
    # x_parallel = cos(theta)*base  => shape (B, M, D)
    x_par = dot_val * base  # broadcast
    # x_perp = x - x_par
    x_perp = x - x_par
    
    scale = theta / (sin_theta + eps)   # (B, M, 1)
    log_vec = scale * x_perp           # (B, M, D)
    
    # theta ~ 0 -> log_vec = 0 처리
    log_vec[mask.squeeze(-1)] = 0.0
    
    return log_vec


def exp_map_batch(base, v, eps=1e-7):
    """
    base: (B, 1, D)
    v   : (B, M, D)
    반환: (B, M, D)
    """
    # norm of v
    phi = v.norm(dim=-1, keepdim=True)   # (B, M, 1)
    mask = (phi < eps)
    
    direction = v / (phi + eps)         # (B, M, D)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    
    # exp_vec = cos(phi)*base + sin(phi)*(v / phi)
    # base shape: (B,1,D) -> broadcast w/ (B,M,D)
    exp_vec = cos_phi * base + sin_phi * direction
    
    # phi ~ 0 => 그대로 base
    exp_vec[mask.squeeze(-1)] = base[mask.squeeze(-1)]
    
    # 마지막에 float 오차 방지 위해 정규화
    exp_vec = F.normalize(exp_vec, dim=-1)
    return exp_vec

def maybe_dictionarize_batch(batch):
    if isinstance(batch, dict):
        return batch
    if len(batch) == 2:
        return {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        return {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

#def test_with_aggregated_coef(model, dataset, base_model=None, merge_learner=None, w1=None, w2=None, merge_depth=13, coef_path='', rep_mask_depth=0, instance_wise=0, runname=0, modelname='ViT-B/32',clip_flag=True, computemetrics=False):

def test_model_on_dataset(model, dataset, base_model=None, merge_learner=None, w1=None, w2=None, merge_depth=13, coef_path='', rep_mask_depth=0, instance_wise=0, runname=0, modelname='ViT-B/32', clip_flag=True, computemetrics=False, ttagg=0, half_prec=0):
    if model is not None:
        model.eval()
    device = 'cuda'
    with torch.no_grad():
        top1, top5, correct1, correct5, n = 0., 0., 0., 0., 0.
        end = time.time()
        loader = dataset.test_loader
        coefs = torch.tensor([]).to(device)
        dname = type(dataset).__name__
        if dname == 'ImageNet2p':
            loader = dataset.train_loader
            assert dataset.train_dataset.__getitem__(dataset.sampler.indices[1000])['image_paths'].endswith('n01675722_4108.JPEG')

        for i, batch in enumerate(tqdm(loader)):
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].cuda(), batch['labels'].cuda()
            if half_prec:
                inputs = inputs.half()

            y = labels
            logits, features = model(inputs, return_features=True)

            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            if isinstance(logits, list):
                logits = logits[0]

            pred_top1 = logits.argmax(dim=1, keepdim=True).to(device)
            pred_top5 = logits.topk(5, dim=1).indices.to(device)

            # Top-1 accuracy
            correct_current1 = pred_top1.eq(y.view_as(pred_top1)).sum().item()
            correct1 += correct_current1

            # Top-5 accuracy
            correct_current5 = sum(y[i] in pred_top5[i] for i in range(len(y)))
            correct5 += correct_current5

            n += y.size(0)

            if i % 20 == 0:
                percent_complete = 100.0 * i / len(loader)
                print(
                    f"[{percent_complete:.0f}% {i}/{len(loader)}]\t"
                    f"Top-1 Acc: {100 * (correct1/n):.2f}\t"
                    f"Top-5 Acc: {100 * (correct5/n):.2f}"
                )

        top1 = correct1 / n
        top5 = correct5 / n

        if coef_path:
            if not os.path.isdir(coef_path): os.makedirs(coef_path, exist_ok=True)
            with open(os.path.join(coef_path, f'{type(dataset).__name__}_{runname}.npy'), 'wb') as f:
                np.save(f, np.array(coefs.cpu()))

        return top1, top5



##################################################
# (A) 헬퍼 함수들: ambiguous vector + 투영 로직
##################################################

def project_naive(fvec, avec, alpha=1.0):
    """
    fvec: (D,) - 원본 임베딩 (예: 특정 클래스 임베딩)
    avec: (D,) - ambiguous vector (평균 임베딩)
    alpha: float
    """
    # 단위화(필요 시)
    Ahat = avec / (avec.norm() + 1e-7)
    dotval = torch.dot(fvec, Ahat)          # scalar
    proj = dotval * Ahat                    # fvec의 A방향 성분
    orth = fvec - proj                      # fvec의 A에 수직인 성분
    # alpha=1 -> 원본 그대로 (proj + orth = fvec)
    fvec_prime = proj + alpha * orth
    return fvec_prime


def project_geodesic(fvec, avec, alpha=1.0):
    """
    fvec, avec는 이미 L2 정규화된 상태라 가정
    alpha=0 -> fvec (원본),
    alpha=1 -> avec (완전히 A 쪽)
    """
    # 혹시 모를 수치 문제 방지
    f_norm = fvec.norm() + 1e-7
    a_norm = avec.norm() + 1e-7

    Fhat = fvec / f_norm
    Ahat = avec / a_norm
    cos_theta = torch.clamp(torch.dot(Fhat, Ahat), -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    if theta < 1e-6:
        # 거의 같은 방향이면 그대로
        return fvec
    sin_theta = torch.sin(theta)

    # SLERP
    f_factor = torch.sin((1-alpha)*theta) / sin_theta
    a_factor = torch.sin(alpha*theta) / sin_theta
    fvec_prime = f_factor * Fhat + a_factor * Ahat
    # 길이를 1로 맞추고 싶으면 아래처럼:
    # fvec_prime = fvec_prime / (fvec_prime.norm() + 1e-7)
    return fvec_prime


##################################################
# (B) Ambiguous Vector 투영을 적용한 test 함수
##################################################

def test_model_on_dataset_ambig(
    model, 
    dataset,
    zeroshot_weights,            # (num_classes, D) 형태의 텍스트 임베딩
    alpha=1.0,
    projection='naive',          # or 'geodesic'
    half_prec=0
):
    """
    기존 test_model_on_dataset와 유사한 구조이되,
    1) 첫 pass에서 top-5 예측
    2) 각 샘플별로 ambiguous vector A 계산
    3) 텍스트 임베딩을 투영(naive / geodesic + alpha)
    4) 재계산한 logits으로 최종 accuracy 산출
    """
    model.eval()
    device = 'cuda'
    loader = dataset.test_loader

    
    if half_prec:
        zeroshot_weights = zeroshot_weights.half()
    # num_classes, dimension
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
            # 여기서 'logits_1st'는 이미 model이 zeroshot_weights dot-product를 했을 수도 있고,
            # 혹은 standard FC 분류자를 통과했을 수도 있음.
            # 만약 "image_features"만 얻고 직접 dot(zeroshot_weights.T) 하고 싶다면,
            #  -> logits_1st = features @ zeroshot_weights.T
            # 라고 수정할 수도 있음 (모델 구조에 따라).

            # 일단 model이 반환한 logits_1st에서 top-5를 구함
            top5_1st = logits_1st.topk(5, dim=1).indices  # shape: (batch_size, 5)

        # -----------------------------
        # 2) "ambiguous vector" 기반 재계산
        #    각 샘플별 top-5 클래스 -> A 구하고,
        #    zeroshot_weights[c]를 투영 후 dot-product
        # -----------------------------
        # (batch_size, D)
        # 만약 features가 L2 normalized가 안 돼 있으면 normalize
        image_emb = torch.nn.functional.normalize(features, dim=1)

        # adjusted_logits: (batch_size, num_classes)
        adjusted_logits = torch.zeros(inputs.size(0), num_classes, device=device)

        # 샘플별로 처리
        for bidx in range(inputs.size(0)):
            # top-5 클래스
            c_top5 = top5_1st[bidx]  # shape (5,)
            # image 임베딩
            f_b = image_emb[bidx]    # shape (D,)

            # 모든 클래스에 대한 로짓을 구해야 최종 top-1 / top-5 뽑을 수 있음
            # -> 실제 질문 내용상, "top-5 내 클래스들끼리만 A를 만든다" 라고 했으므로
            #    c_top5끼리만 상호 보완. (나머지는 원본 임베딩 그대로?)
            #    여기서는 예시로,
            #    "top-5 내에 있는 클래스 c"는 ambiguous vector를 (다른 top-5 클래스들의 평균)으로 잡아 투영
            #    "top-5 바깥 클래스"는 굳이 안 만짐(=원본)
            # -------------------------------------------------------

            # 먼저 "top-5 바깥"인 클래스들의 로짓 계산(원본 임베딩 사용)
            #  shape: (num_classes, D)
            #  image_emb (D,) @ w_c (D,) -> scalar
            #  -> 여기서는 한번에 계산: (1, D) @ (num_classes, D).T = (1, num_classes)
            #  단, top-5 내 클래스들은 곧 따로 업데이트할 것이므로 복사만 해둠
            naive_logits_b = f_b.unsqueeze(0) @ zeroshot_weights.float().transpose(0, 1)  # (1, num_classes)
            naive_logits_b = naive_logits_b.squeeze(0)  # (num_classes,)

            # 우선 전체를 naive_logits_b로 초기화
            adjusted_logits[bidx] = naive_logits_b

            # 이제 "top-5 안에 있는 클래스들"을 투영하여 다시 계산
            c_top5_list = c_top5.tolist()
            for c_idx in c_top5_list:
                # 2-1) ambiguous vector A = 평균(다른 top-5 클래스 임베딩)
                others = [x for x in c_top5_list if x != c_idx]
                if len(others) == 0:
                    # top-5가 자기 자신만인 경우(이론상 거의 불가능)
                    A = zeroshot_weights[c_idx].clone()
                else:
                    A = zeroshot_weights[others, :].mean(dim=0)
                    A = A / (A.norm() + 1e-7)

                # 2-2) 원본 텍스트 임베딩 F
                F = zeroshot_weights[c_idx]
                
                # 2-3) 투영
                if projection == 'naive':
                    Fprime = project_naive(F, A, alpha=alpha).float()
                else:
                    # geodesic
                    # geodesic의 경우 fvec, avec가 이미 L2 정규화된 걸 권장
                    F_norm = F / (F.norm() + 1e-7)
                    A_norm = A / (A.norm() + 1e-7)
                    # alpha=0이 "원본 F"가 되도록 하고 싶다면 조금 조정이 필요
                    # 일단은 그대로 사용
                    Fprime = project_geodesic(F_norm, A_norm, alpha=alpha).float()
                
                # 2-4) 새로운 로짓
                #     (image 임베딩 f_b) dot (Fprime)
                new_logit = torch.dot(f_b, Fprime)
                adjusted_logits[bidx, c_idx] = new_logit

        # -----------------------------
        # 3) 최종 top-1 / top-5 정확도 계산
        # -----------------------------
        # 데이터셋이 project_logits 등을 쓸 수도 있으니 체크
        projection_fn = getattr(dataset, 'project_logits', None)
        if projection_fn is not None:
            adjusted_logits = projection_fn(adjusted_logits, device)

        if hasattr(dataset, 'project_labels'):
            labels_ = dataset.project_labels(labels, device)
        else:
            labels_ = labels

        pred_top1 = adjusted_logits.argmax(dim=1, keepdim=True)
        pred_top5 = adjusted_logits.topk(5, dim=1).indices

        # Top-1 accuracy
        correct_current1 = pred_top1.eq(labels_.view_as(pred_top1)).sum().item()
        correct1 += correct_current1

        # Top-5 accuracy
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
    print(f"Done. Final Top-1: {100*top1:.2f}%, Top-5: {100*top5:.2f}%, Time: {elapsed:.1f}s")

    return top1, top5
# original function
# def test_model_on_dataset(model, dataset, base_model=None, merge_learner=None, w1=None, w2=None, merge_depth=13, coef_path='', rep_mask_depth=0, instance_wise=0, runname=0, modelname='ViT-B/32',clip_flag=True, computemetrics=False,ttagg=0,half_prec=0):
#     if model is not None:
#         model.eval()
#     device = 'cuda'
#     with torch.no_grad():
#         top1, correct, n = 0., 0., 0.
#         end = time.time()
#         loader = dataset.test_loader
#         coefs = torch.tensor([]).to(device)
#         dname = type(dataset).__name__
#         if dname == 'ImageNet2p':
#             loader = dataset.train_loader
#             # assert to make sure the imagenet held-out minival logic is consistent across machines.
#             # tested on a few machines but if this fails for you please submit an issue and we will resolve.
#             assert dataset.train_dataset.__getitem__(dataset.sampler.indices[1000])['image_paths'].endswith('n01675722_4108.JPEG')

#         for i, batch in enumerate(tqdm(loader)):
#             batch = maybe_dictionarize_batch(batch)
#             # pdb.set_trace()
#             inputs, labels = batch['images'].cuda(), batch['labels'].cuda()
#             if half_prec:
#                 inputs = inputs.half()

#             data_time = time.time() - end
#             y = labels
#             if 'image_paths' in batch:
#                 image_paths = batch['image_paths']

#             if merge_learner is not None:
#                 #print('performing testtime merging')
#                 assert base_model is not None
#                 if rep_mask_depth:
#                     coef, mask, scale = merge_learner(inputs)
#                 else:
#                     mask, scale = None, None
#                     coef = merge_learner(inputs)
                
#                 if instance_wise:
#                     bs = inputs.shape[0]
#                     logits = torch.tensor([]).cuda()
#                     for j in range(bs):
#                         merged_sd = interpolation(coef[j], w1, w2, merge_depth, modelname)
#                         model = get_model_from_sd(merged_sd, base_model, mask, scale, clip_flag=clip_flag)
#                         model.eval()
#                         logit = model(inputs[j].unsqueeze(0))
#                         logits = torch.cat([logits, logit])
#                 else:
#                     merged_sd = interpolation(coef, w1, w2, merge_depth, modelname)
#                     model = get_model_from_sd(merged_sd, base_model, mask, scale, clip_flag=clip_flag)
#                     model.eval()
#                     if half_prec:
#                         convert_weights(model)
                    
#                     logits, features = model(inputs, return_features=True)
#                 coefs = torch.cat([coefs, coef])
#             else:
#                 logits, features = model(inputs, return_features=True)

#             projection_fn = getattr(dataset, 'project_logits', None)
#             if projection_fn is not None:
#                 logits = projection_fn(logits, device)

#             if hasattr(dataset, 'project_labels'):
#                 y = dataset.project_labels(y, device)
#             if isinstance(logits, list):
#                 logits = logits[0]

#             pred = logits.argmax(dim=1, keepdim=True).to(device)
#             if hasattr(dataset, 'accuracy'):
#                 acc1, num_total = dataset.accuracy(logits, y, image_paths, None)
#                 correct += acc1
#                 n += num_total
#             else:
#                 correct_current = pred.eq(y.view_as(pred)).sum().item()
#                 correct += correct_current
#                 n += y.size(0)

#             batch_time = time.time() - end
#             end = time.time()
#             if i % 20 == 0:
#                 percent_complete = 100.0 * i / len(loader)
#                 print(
#                     f"[{percent_complete:.0f}% {i}/{len(loader)}]\t"
#                     f"Acc: {100 * (correct/n):.2f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
#                 )

#         top1 = correct / n
#         # try:
#         if coef_path:
#             if not os.path.isdir(coef_path): os.makedirs(coef_path, exist_ok=True)
#             with open(os.path.join(coef_path,f'{type(dataset).__name__}_{runname}.npy'), 'wb') as f:
#                 np.save(f,np.array(coefs.cpu()))
    
#         return top1

def test_edm_on_dataset(merge_learner, dataset, base_model=None, w1=None, w2=None, merge_depth=13, coef_path='', runname=0, modelname='ViT-B/32',clip_flag=True):
    if merge_learner is not None: merge_learner.eval()
    
    device = 'cuda'
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        end = time.time()
        loader = dataset.test_loader
        coefs = torch.tensor([]).to(device)
        merge_indices = torch.tensor([]).to(device)
        dname = type(dataset).__name__
        if dname == 'ImageNet2p':
            loader = dataset.train_loader
            assert dataset.train_dataset.__getitem__(dataset.sampler.indices[1000])['image_paths'].endswith('n01675722_4108.JPEG')

        for i, batch in enumerate(tqdm(loader)):
            batch = maybe_dictionarize_batch(batch)
            x, y = batch['images'].cuda(), batch['labels'].cuda()
            data_time = time.time() - end
            if 'image_paths' in batch:
                image_paths = batch['image_paths']

            if merge_learner.value_type != 'model':
                raise ValueError('not implemented yet')
            else:
                loss, l_dict, merge_clf_idx, y_recon, logits = merge_learner(x, y, eval_mode=True)
            
            merge_indices = torch.cat([merge_indices, merge_clf_idx])

            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y_recon = dataset.project_labels(y_recon, device)
            if isinstance(logits, list):
                logits = logits[0]

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y_recon, image_paths, None)
                correct += acc1
                n += num_total
            else:
                correct_current = pred.eq(y_recon.view_as(pred)).sum().item()
                correct += correct_current
                n += y_recon.size(0)

            batch_time = time.time() - end
            end = time.time()
            if i % 20 == 0:
                percent_complete = 100.0 * i / len(loader)
                print(
                    f"[{percent_complete:.0f}% {i}/{len(loader)}]\t"
                    f"Acc: {100 * (correct/n):.2f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                )
            
        top1 = correct / n
        npidx = np.array(merge_indices.cpu()).astype(np.int32)

        if coef_path:
            if not os.path.isdir(coef_path): os.makedirs(coef_path, exist_ok=True)
            with open(os.path.join(coef_path,f'{type(dataset).__name__}_indices_{runname}.npy'), 'wb') as f:
                np.save(f,npidx)
        
            print("*** count of estimated merged model indices ***")
            print(np.bincount(npidx))

        return top1, np.bincount(npidx)

def test_wbtrainer_on_dataset(trainer, dataset, loader_input=None, coef_path='', instance_wise=0, runname=0, modelname='ViT-B/32',clip_flag=True, computemetrics=False,ttagg=0, validation=False,half_prec=0):
    device = 'cuda'
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        end = time.time()
        loader = dataset.test_loader if loader_input is None else loader_input
        coefs = torch.tensor([]).to(device)
        dname = type(dataset).__name__
        if dname == 'ImageNet2p':
            loader = dataset.train_loader if loader_input is None else loader_input
            # assert to make sure the imagenet held-out minival logic is consistent across machines.
            # tested on a few machines but if this fails for you please submit an issue and we will resolve.
            # assert dataset.train_dataset.__getitem__(dataset.sampler.indices[1000])['image_paths'].endswith('n01675722_4108.JPEG')
        
        if ttagg:
            assert trainer.ddriven == 1
            print('perform test-time coefficient aggregation')
            trainer.best_learner.instance_wise = 1
            coef = torch.tensor([]).to('cuda')
            for i, batch in enumerate(tqdm(loader)):
                batch = maybe_dictionarize_batch(batch)
                inputs = batch['images'].cuda()
                data_time = time.time() - end
                if 'image_paths' in batch:
                    image_paths = batch['image_paths']
                
                coef = torch.cat((coef,trainer.best_learner(inputs)))
            
            if coef_path:
                print(f'save coef at {coef_path} before aggregation')
                if not os.path.isdir(coef_path): 
                    os.makedirs(coef_path)
                with open(os.path.join(coef_path,f'{type(dataset).__name__}_{runname}_beforeagg.npy'), 'wb') as f:
                    np.save(f,np.array(coef.cpu()))
                    
            print(f'aggregate {coef.shape[0]} coefs to single coef')
            agg_coef = coef.mean(0)

            # merged_sd = interpolation(agg_coef, w1, w2, merge_depth, modelname)
            # model = get_model_from_sd(merged_sd, base_model, mask, scale, clip_flag=clip_flag)
            model = trainer.get_merged_model(agg_coef)
            model.eval()
            merge_learner, mask, scale = None, None, None

        for i, batch in enumerate(tqdm(loader)):
            batch = maybe_dictionarize_batch(batch)
            # pdb.set_trace()
            inputs, labels, path = batch['images'].cuda(), batch['labels'].cuda(), batch['image_paths']
            if half_prec:
                inputs = inputs.half()
            data_time = time.time() - end
            y = labels
            if 'image_paths' in batch:
                image_paths = batch['image_paths']
            if trainer.ddriven:
                if validation:
                    logits, _, coef = trainer(inputs, use_best=False, out_coef=True, path=path[0])
                else:
                    logits, _, coef = trainer(inputs, use_best=True, out_coef=True, path=path[0])
                coefs = torch.cat([coefs, coef])
            else:
                if validation:
                    logits = trainer.model(inputs)
                else:
                    logits = trainer.best_model(inputs)

            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            if isinstance(logits, list):
                logits = logits[0]

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths, None)
                correct += acc1
                n += num_total
            else:
                correct_current = pred.eq(y.view_as(pred)).sum().item()
                correct += correct_current
                n += y.size(0)

            batch_time = time.time() - end
            end = time.time()
            if i % 20 == 0:
                percent_complete = 100.0 * i / len(loader)
                print(
                    f"[{percent_complete:.0f}% {i}/{len(loader)}]\t"
                    f"Acc: {100 * (correct/n):.2f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                )
            #if i > 2: break
        top1 = correct / n

        if coef_path:
            if not os.path.isdir(coef_path): os.makedirs(coef_path, exist_ok=True)
            with open(os.path.join(coef_path,f'{type(dataset).__name__}_{runname}.npy'), 'wb') as f:
                np.save(f,np.array(coefs.cpu()))
        
        return top1

def test_wbtrainer_assigner_on_dataset(trainer, dataset, loader_input=None, coef_path='', runname=0, modelname='ViT-B/32',clip_flag=True, validation=False):
    device = 'cuda'
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        end = time.time()
        loader = dataset.test_loader if loader_input is None else loader_input
        idxs = torch.tensor([]).to(device)
        dname = type(dataset).__name__
        if dname == 'ImageNet2p':
            loader = dataset.train_loader if loader_input is None else loader_input
        
        for i, batch in enumerate(tqdm(loader)):
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].cuda(), batch['labels'].cuda()
            data_time = time.time() - end
            y = labels
            if 'image_paths' in batch:
                image_paths = batch['image_paths']
            
            if validation:
                _, _, idx, y_recon, logits = trainer(inputs, y, use_best=False, eval_mode=True)
            else:
                _, _, idx, y_recon, logits = trainer(inputs, y, use_best=True, eval_mode=True)
            idxs = torch.cat([idxs, idx])
  
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y_recon = dataset.project_labels(y_recon, device)
            if isinstance(logits, list):
                logits = logits[0]

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y_recon, image_paths, None)
                correct += acc1
                n += num_total
            else:
                correct_current = pred.eq(y_recon.view_as(pred)).sum().item()
                correct += correct_current
                n += y_recon.size(0)

            batch_time = time.time() - end
            end = time.time()
            if i % 20 == 0:
                percent_complete = 100.0 * i / len(loader)
                print(
                    f"[{percent_complete:.0f}% {i}/{len(loader)}]\t"
                    f"Acc: {100 * (correct/n):.2f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                )
        
        top1 = correct / n
        npidx = np.array(idxs.cpu()).astype(np.int32)

        if coef_path:
            if not os.path.isdir(coef_path): os.makedirs(coef_path, exist_ok=True)
            with open(os.path.join(coef_path,f'{type(dataset).__name__}_indices_{runname}.npy'), 'wb') as f:
                np.save(f,npidx)
        
            print("*** count of estimated merged model indices ***")
            print(np.bincount(npidx))

        return top1, np.bincount(npidx)

def clip_clipping(x):
    #! -inf ~ inf -> CLIP's input RGB range
    if len(x.shape) == 3:
        out = torch.cat([torch.clip(x[0,:,:], min=-1.79226253, max=1.93033625).unsqueeze(0),
                     torch.clip(x[1,:,:], min=-1.75209713, max=2.07488384).unsqueeze(0),
                     torch.clip(x[2,:,:], min=-1.48021977, max=2.14589699).unsqueeze(0)], dim=0)
    else:
        out = torch.cat([torch.clip(x[:,0,:,:], min=-1.79226253, max=1.93033625).unsqueeze(1),
                        torch.clip(x[:,1,:,:], min=-1.75209713, max=2.07488384).unsqueeze(1),
                        torch.clip(x[:,2,:,:], min=-1.48021977, max=2.14589699).unsqueeze(1)], dim=1)
    return out

def perturb_generator(data, max_noise=2.0):
    perturbed_data = clip_clipping(data + max_noise * torch.randn(data.size(), device=data.device))
    return perturbed_data

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

def _convert_image_to_rgb(image):
    return image.convert("RGB")
    
def get_simclr_pipeline_transform(size=224, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * size)),
                                            _convert_image_to_rgb,
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
    # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    # data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size, scale=(0.5, 1.0)),
    #                                         transforms.RandomHorizontalFlip(),
    #                                         transforms.RandomApply([color_jitter], p=0.8),
    #                                         transforms.RandomGrayscale(p=0.2),
    #                                         GaussianBlur(kernel_size=int(0.1 * size)),
    #                                         _convert_image_to_rgb,
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
    # color_jitter = transforms.ColorJitter(0.2 * s, 0.2 * s, 0.2 * s, 0.2 * s)
    # data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
    #                                         transforms.RandomHorizontalFlip(),
    #                                         transforms.RandomApply([color_jitter], p=0.4),
    #                                         transforms.RandomGrayscale(p=0.2),
    #                                         GaussianBlur(kernel_size=int(0.1 * size)),
    #                                         _convert_image_to_rgb,
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
    return data_transforms

#* CLIP transformation
# def _transform(n_px):
#     return Compose([
#         Resize(n_px, interpolation=BICUBIC),
#         CenterCrop(n_px),
#         _convert_image_to_rgb,
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])

def identity(x,device='cuda'):
    return x

def compute_sample_loss_uncertainty(logits, labels, temperature=1.0,pseudo_label=0,prob=False):
    if pseudo_label in [1,2]:
        loss = (-labels * F.log_softmax(logits,1)).sum(1)
        #pdb.set_trace()
    else:
        loss = F.cross_entropy(logits, labels, reduction='none')

    if prob:
        uncertainty = - (logits * torch.log(logits)).sum(dim=1)
    else:
        uncertainty = - (F.softmax(logits/temperature,dim=1) * F.log_softmax(logits/temperature, dim=1)).sum(dim=1)
    return loss.reshape(-1,), uncertainty.reshape(-1,)

def compute_accuracy(logits, labels):
    pred = logits.argmax(dim=1, keepdim=True)
    correct = pred.eq(labels.view_as(pred)).sum().item()
    n = labels.size(0)
    acc = correct / n

    return acc

def compute_sample_accuracy(logits, labels):
    pred = logits.argmax(dim=1, keepdim=True)
    return pred.eq(labels.view_as(pred))

def compute_sample_uncertainty(logits, temperature=1.0):
    uncertainty = - (F.softmax(logits/temperature,dim=1) * F.log_softmax(logits/temperature, dim=1)).sum(dim=1)
    return uncertainty.reshape(-1,)

def compute_sample_loss_uncertainty_acc(logits, labels, temperature=1.0, pseudo_label=0):
    if pseudo_label in [1,2]:
        loss = (-labels * F.log_softmax(logits,1)).sum(1)
        #pdb.set_trace()
    else:
        loss = F.cross_entropy(logits, labels, reduction='none')
    
    pred = logits.argmax(dim=1, keepdim=True)

    uncertainty = - (F.softmax(logits/temperature,dim=1) * F.log_softmax(logits/temperature, dim=1)).sum(dim=1)
    return loss.reshape(-1,), pred.eq(labels.view_as(pred)), uncertainty.reshape(-1,)

def get_feats_logits_and_labels(model, preprocess, args, dataset_name=None,state_dict=None):
    dataset_name = args.source_dataset if dataset_name is None else dataset_name
    aug = ''
    if dataset_name == 'ImageNet2p':
        if args.data_aug != 'noaug':
            if args.auto_aug is not None:
                aug = 'rand'
            else:
                aug = args.data_aug
    
    if args.model == 'ViT-B/32': pass
    else:
        if args.model == 'ViT-B/16': mpf = 'CLIPVIT_B16_'
        if args.model == 'ViT-L/14': mpf = 'CLIPVIT_L14_'
        if 'eva' in args.model:      mpf = 'EVA02_'
        args.prefix = mpf + args.prefix
    
    if args.half_prec:
        args.prefix = args.prefix + 'fp16_'
    else:
        pass

    save_path = os.path.join(args.cache_dir, args.prefix + dataset_name + aug + '_cache.pt')
    device = args.device
    
    prefix = args.prefix

    dataset_class = getattr(datasets, dataset_name)
    dataset = dataset_class(
        preprocess,
        args.data_location,
        args.batch_size,
        args.workers
    )
    project_fn = getattr(dataset, 'project_logits', None)
    dataloader = dataset.test_loader
    if dataset_name == 'ImageNet2p':
        dataloader = dataset.train_loader

    if (os.path.exists(save_path)) and (args.nosave != 1):
        cache_data = torch.load(save_path)
        logits = cache_data['logits'].to(device)
        labels = cache_data['labels'].to(device)
        feats = cache_data['feats'].to(device)
    else:
        print(f"do not find cache in {save_path}")
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
        
        if state_dict is not None:
            model = get_model_from_sd(state_dict, model)

        model.eval()
        logits, labels, feats = [], [], []
        top1, correct, n = 0., 0., 0.
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                data = maybe_dictionarize_batch(data)
                x = data['images'].to(device)
                if args.half_prec:
                    #pdb.set_trace()
                    x = x.half()
                    #x = x.type(torch.HalfTensor)

                label = data['labels'].to(device)
                if 'image_paths' in data:
                    image_paths = data['image_paths']
                logit, feat = model(x, return_features=True)

                labels.append(label)
                logits.append(logit)
                feats.append(feat)

                if project_fn is not None:
                    logit = project_fn(logit, device)
                if hasattr(dataset, 'project_labels'):
                    label = dataset.project_labels(label, device)
                if isinstance(logit, list):
                    logit = logit[0]
                pred = logit.argmax(dim=1, keepdim=True).to(device)
                # if hasattr(dataset, 'accuracy'):
                #     acc1, num_total = dataset.accuracy(logit, label, image_paths, None)
                #     correct += acc1
                #     n += num_total
                # else:
                #     correct += pred.eq(label.view_as(pred)).sum().item()
                #     n += label.size(0)
                    
        labels = torch.cat(labels)
        logits = torch.cat(logits)
        #feats = torch.cat(feats)

        if args.nosave:
            pass
        else:
            print(f'successfully save cache files at {save_path}')
            torch.save({'logits': logits.cpu(), 'labels': labels.cpu(), 'feats': feats.cpu()}, save_path)

        #print(f"{dataset_name}_{prefix} Acc: {correct/n:.3f}")
    if project_fn is not None:
        return feats, logits, labels, project_fn
    else:
        return feats, logits, labels, identity

def get_feats_logits(model, args, dataloader, project_fn):
    model.eval()
    logits = []
    top1, correct, n = 0., 0., 0.
    device = args.device
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            data = maybe_dictionarize_batch(data)
            x = data['images'].to(device)
            if args.half_prec:
                x = x.half()

            if 'image_paths' in data:
                image_paths = data['image_paths']
            #pdb.set_trace()
            logit, _ = model(x, return_features=True)

            if project_fn is not None:
                logit = project_fn(logit, device)
            if isinstance(logit, list):
                logit = logit[0]
            pred = logit.argmax(dim=1, keepdim=True).to(device)
            logits.append(logit)

        logits = torch.cat(logits)
        
    if project_fn is not None:
        return logits, project_fn
    else:
        return logits, identity

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class RandomGaussianDataset(Dataset):
    def __init__(self,
                 mean1=[0.48145466, 0.4578275, 0.40821073],
                 std1=[0.26862954, 0.26130258, 0.27577711],
                 n_samples=50000,
                 image_size=(3,224,224),
                 transform=None):
        """
        Args:
            mean1 (float): Mean of the Gaussian distribution for class 0.
            std1 (float): Standard deviation of the Gaussian distribution for class 0.
            mean2 (float): Mean of the Gaussian distribution for class 1.
            std2 (float): Standard deviation of the Gaussian distribution for class 1.
            n_samples (int): Total number of samples (for both classes combined).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.mean1 = torch.tensor(mean1)
        self.std1 = torch.tensor(std1)
        # self.mean2 = mean2
        # self.std2 = std2
        self.n_samples = n_samples
        self.transform = transform
        self.image_size = image_size
        
        # Generate data
        self.data, self.labels = self._generate_data()
    
    def _generate_data(self):
        # half_samples = self.n_samples // 2
        
        # # Generate class 0 samples
        # class0_data = torch.randn(half_samples) * self.std1 + self.mean1
        # class0_labels = torch.zeros(half_samples)
        
        # # Generate class 1 samples
        # class1_data = torch.randn(half_samples) * self.std2 + self.mean2
        # class1_labels = torch.ones(half_samples)
        
        # Combine and shuffle data
        # data = torch.cat([class0_data, class1_data])
        # labels = torch.cat([class0_labels, class1_labels])

        # indices = torch.randperm(self.n_samples)
        # data = data[indices]
        # labels = labels[indices]
        ## pdb.set_trace()
        data = torch.randn(self.n_samples, *self.image_size) * self.std1.reshape(1,3,1,1) + self.mean1.reshape(1,3,1,1)
        labels = torch.zeros(self.n_samples) 
        
        return data, labels
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return {'images':sample, 'labels':label} #sample, label
    
def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


def mixup_data(x, y, beta=0.8, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if beta > 0:
        lam = np.random.beta(beta, beta)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh

def cutmix_bbox_and_lam(img_shape, lam, correct_lam=True, count=None):
    """ Generate bbox and apply lambda correction.
    """
    yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    
    return (yl, yu, xl, xu), lam

def cutmix_data(x, y, beta=1.0, device='cpu'):
    if beta > 0:
        lam = np.random.beta(beta, beta)
    else:
        lam = 1    
    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
        x.shape, lam)
    x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]    
    y_a, y_b = y, y.flip(0)
    return x, y_a, y_b, lam



def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def test_in(model, args, dataset_lt, preprocess):
    # Note: ImageNet2p is the held-out minival set from ImageNet train that we use.
    # It is called 2p for 2 percent of ImageNet, or 26k images.
    # See utils on how this dataset is handled slightly differently.
    results = {'name': args.name}

    for dataset_cls in dataset_lt:
        print(f'Evaluating model on {dataset_cls.__name__}.')

        dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
        accuracy = test_model_on_dataset(model, dataset)
        results[dataset_cls.__name__] = accuracy
        print(accuracy)
        
    return results

import time
import numpy as np
from scipy.stats import beta
from scipy.special import betaln, gammaln, digamma, logsumexp
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

class MVBetaMM:
    def __init__(self, n_mixtures=1, verbose=False, verbose_interval=10, random_state=1):
        """
        Initializes multivariate beta mixture model. It assumes multivariate via indepent beta distributions
        within mixtures, not via a Dirichlet distribution. This allows a sum > 1.

        Parameters:
        - n_mixtures (int): Number of MvBeta distributions in the model
        - verbose (boolean): If true, will print information during training
        - verbose_interval (int): Amount of iterations between verbose statements
        - random_state (int): Random state for all algorithms
        """
        self.n_mixtures = n_mixtures
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.random_state = random_state
        self.converged = False
    

    def _initialize(self, X, init_num):
        """
        Initializes the model parameters based on the given method

        Parameters:
        - X (matrix): Data to initialize with
        - init_num (int): Initialization number. Updates the random_state
        """
        # Initialize responsibilities based on method
        if self.method == "kmeans":
            resp = np.zeros(shape=(self.n_observations, self.n_mixtures))
            label = KMeans(n_clusters=self.n_mixtures, n_init=1, random_state=(self.random_state + init_num)).fit(X).labels_
            resp[np.arange(self.n_observations), label] = 1  # Set responsibility to 1 for the cluster it was assigned
        elif self.method == "random":
            np.random.seed(self.random_state + init_num)
            resp = np.random.uniform(size=(self.n_observations, self.n_mixtures))
            resp /= resp.sum(axis=1)[:, np.newaxis]
        
        # Add a small number for numerical stability (no log(0))
        resp += 10 * np.finfo(resp.dtype).eps

        # Compute the weights, alphas, and betas via M step
        self.params_ = np.zeros((self.n_mixtures, self.n_components*2))
        self._m_step(X, np.log(resp))
        self._verbose_initialization(init_num)

    
    def _estimate_log_weights(self):
        """
        Computes the log weights of the current model

        Returns:
        - log_weights (vector): Natural logarithm of the model weights
        """
        return np.log(self.weights_)
    
    
    def _compute_log_prob_for_mixture(self, X, mix):
        """
        Helper function to compute the log probability for a single mixture. Used for parallel computing

        Parameters:
        - X (matrix): Data
        - mix (int): Mixture number to assess the log probability of

        Returns:
        - log_prob (vector): Log probabilities of each observation associated with this mixture
        """
        alpha = self.params_[mix, :self.n_components]
        beta = self.params_[mix, self.n_components:]

        # Compute the log of the Beta function for each mixture
        log_beta_fn = betaln(alpha, beta)

        # Compute the log probability for each observation for current mixture
        log_prob = ((alpha - 1) * np.log(X) + (beta - 1) * np.log(1 - X) - log_beta_fn).sum(axis=1)
        return log_prob

    def _estimate_log_prob(self, X):
        """
        Estimates the log probability for all the mixtures

        Parameters:
        - X (matrix): Data

        Returns:
        - log_prob (matrix): Matrix of log probabilities. ij entry is the (unnormalized) log probability that 
                             observation i belongs to mixture j
        """
        # Don't use parallel computing at all when n_jobs=1. The initialization cost of parallel computing is high even for n_jobs=1
        if self.n_jobs == 1:
            log_prob = np.empty((self.n_observations, self.n_mixtures))
            for mix in range(self.n_mixtures):
                alpha = self.params_[mix, :self.n_components]
                beta = self.params_[mix, self.n_components:]

                # Compute the log of the Beta function for each mixture
                log_beta_fn = betaln(alpha, beta)

                # Compute the log probability for each observation for current mixture
                log_prob[:, mix] = ((alpha - 1) * np.log(X) + (beta - 1) * np.log(1 - X) - log_beta_fn).sum(axis=1)

            return log_prob

        else:
            log_prob = Parallel(n_jobs=self.n_jobs)(delayed(self._compute_log_prob_for_mixture)(X, mix) for mix in range(self.n_mixtures))
            return np.array(log_prob).T  # Transpose since the helper returns them as row vectors

    
    def _estimate_weighted_log_prob(self, X):
        """
        Estimates the weighted log probabilities for all the mixtures
        
        Parameters:
        - X (matrix): Data

        Returns:
        - weighted_log_prob (matrix): Matrix of weighted log probabilities. ij entry is the weighted (unnormalizd) 
                                      log probability that observation i belongs to mixture j
        """
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    
    def _estimate_log_prob_resp(self, X):
        """
        Estimates the normalized log probabilites and the log responsiblities of each mixture

        Parameters:
        - X (matrix): Data

        Returns:
        - log_prob_norm (vector): Normalizing constant for each observation
        - log_resp (matrix): Matrix of log responsibilities. ij entry is the log prob that 
                             obs i belongs to mixture j (normalized and weighted)
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)  # Normalizing constant

        # Ignore Underflow
        with np.errstate(under="ignore"):
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        
        return log_prob_norm, log_resp
    

    def _e_step(self, X):
        """
        Performs the expectation step of the EM algorithm

        Parameters:
        - X (matrix): Data
        
        Returns:
        - mean_log_prob_norm (matrix): Mean normalizing constant for all the observations
        - log_resp (matrix): Matrix of log responsibilities. ij entry is the log prob that 
                             obs i belongs to mixture j (normalized and weighted)
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp
    

    def exp_responsibilities(self, log_resp):
        """
        Exponentiate the log responsibilities and compute the weighted importance of each mixture (unnormalized)

        Parameters:
        - log_resp (matrix): Matrix of log responsibilities. ij entry is the log prob that 
                             obs i belongs to mixture j (normalized and weighted)
        
        Returns:
        - resp (matrix): Exponentiated log responsibilites matrix
        - nk (vector): Weighted importance of each mixture
        """
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps  # Number of elements in mixture k
        return resp, nk
    

    def update_weights(self, nk):
        """
        Updates the weights of the mixtures

        Parameters:
        - nk (vector): The sum of the probabilities of each mixture over every observation
        """
        self.weights_ = nk / np.sum(nk)
    

    def _m_step(self, X, log_resp):
        """
        Performs the M step of the EM algorithm via 1st and 2nd moment matching. Automatically
        updates the parameters of the model

        Parameters:
        - X (matrix): Data
        - log_resp (matrix): Matrix of log responsibilities. ij entry is the log prob that 
                             obs i belongs to mixture j (normalized and weighted)
        """

        # Update the weights
        resp, nk = self.exp_responsibilities(log_resp)
        self.update_weights(nk)
        
        # Calculate weighted sums and square sums for each mixture
        weighted_sums = resp.T @ X
        weighted_square_sums = resp.T @ (X ** 2)

        for i in range(self.n_mixtures):
            # Get weighted sum and square sum for this mixture
            weighted_sum = weighted_sums[i]
            weighted_square_sum = weighted_square_sums[i]
            
            # Calculate weighted mean and variance for each feature
            weighted_mean = weighted_sum / nk[i]
            weighted_variance = weighted_square_sum / nk[i] - weighted_mean ** 2

            # Compute the maximum possible weighted variance
            max_possible_weighted_variance = weighted_mean * (1 - weighted_mean) / 4
            weighted_variance = np.minimum(weighted_variance, max_possible_weighted_variance)
            weighted_variance += 10 * np.finfo(weighted_variance.dtype).eps

            # Calculate common factor once for each mixture
            common_factor = weighted_mean * (1 - weighted_mean) / (weighted_variance + 1e-10) - 1

            # Update parameters
            self.params_[i, :self.n_components] = common_factor * weighted_mean  # alphas
            self.params_[i, self.n_components:] = common_factor * (1 - weighted_mean)  # betas


    def _verbose_initialization(self, n):
        if self.verbose:
            print(f"New {self.method} initialization. Init Number {n + 1}")


    def _verbose_iter(self, iter, lower_bound, avg_time):
        if self.verbose and iter % self.verbose_interval == 0:
            print(f"Training Iteration {iter} complete. Current log prob lower bound: {lower_bound}. Avg Training Time: {np.round(avg_time, decimals=1)}s")

    
    def _verbose_converged(self, iter, lower_bound):
        if self.verbose:
            print(f"Converged on iteration {iter}. Log probability lower bound: {lower_bound}")


    def fit(self, X, n_init=3, method="kmeans", max_iter=200, tol=1e-4, n_jobs=8):
        """
        Fits the parameters and weights of the MVBeta model to maximize the loglikelihood of the model
        given the data X.

        Parameters:
        - X (matrix): Data to fit the model to
        - n_init (int): Number of initializations to try
        - max_iter (int): Maximum number of iterations allowed if convergence is not yet reached
        - tol (float): minimum allowed difference in log likelihoods before convergence is reached
        - n_jobs (int): Number of CPU cores to use on the E-Step (can significantly speed up compute)

        Returns:
        - self
        """
        self.n_observations, self.n_components = X.shape

        self.n_jobs = n_jobs
        self.converged = False
        max_lower_bound = -np.inf
        self.method = method if method.lower() in ["kmeans", "random"] else "kmeans"

        for init in range(n_init):
            print(f'{init+1}-th BMM initialization')
            self._initialize(X, init)
            lower_bound = -np.inf

            start = time.time()
            for iter in tqdm(range(max_iter)):
                prev_lower_bound = lower_bound
                
                # Used to print average iter duration
                if iter % self.verbose_interval == 0:
                    start = time.time()
                    start_iter = iter

                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)

                lower_bound = log_prob_norm
                change = lower_bound - prev_lower_bound

                end = time.time()
                avg_time = (end - start)/(iter - start_iter + 1)

                self._verbose_iter(iter, lower_bound, avg_time)

                if abs(change) < tol:
                    self._verbose_converged(iter, lower_bound)
                    self.converged = True
                    break
            
            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound

                # Update the weights again to reflect the weights with the new parameters
                _, nk = self.exp_responsibilities(log_resp)
                self.update_weights(nk)

                best_params = [self.weights_, self.params_]
        
        self.weights_ = best_params[0]
        self.params_ = best_params[1]
        self.max_lower_bound = max_lower_bound
        
        return self
    

    def set_verbose(self, verbose, interval):
        """
        Update the verbose parameters

        Parameters:
        - verbose (boolean): If true, prints updates every interval iters
        - interval (int): Frequency of verbose statements
        """
        self.verbose = verbose
        self.verbose_interval = interval

    
    def set_jobs(self, n_jobs):
        """
        Updates n_jobs

        Parameters:
        - n_jobs (int): Number of CPUs to use in training
        """
        self.n_jobs = n_jobs
    

    def predict_proba(self, X):
        """
        Predcits the probability that X belongs to each of the distributions

        Parameters:
        - X (matrix): Data to probabilistically evaluate

        Returns:
        - Probs (matrix): NxK matrix where K is number of mixtures. ij is the probability obs i belongs to mixture k
        """
        _, log_resp = self._e_step(X)
        return np.exp(log_resp)
    

    def predict(self, X):
        """
        Predicts the most likely distribution for each observation in X

        Parameters:
        - X (matrix): Data to predict classes of

        Returns:
        - Classes (vector): The predicted classes [0, K-1] of the observations
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    

    def show_info(self):
        """
        Shows relevant information about the model. Includes the number of mixtures, training set info, and the status of model convergence
        """
        print(self)


    def _n_parameters(self):
        """
        Returns the number of free parameters in the current model
        """
        # Minus 1 since the last weight = 1 - sum(other weights)
        return int(2 * self.n_components + self.n_mixtures - 1)

    
    def score_samples(self, X):
        """
        Compute the log likelihood of each sample

        Parameters:
        - X (matrix): Data

        Returns:
        - log_prob: Log likelihood of each sample under the model
        """

        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)
    

    def score(self, X):
        """
        Compute average log likelihood over all samples

        Parameters:
        - X (matrix): Data

        Returns:
        - avg_log_prob: Average log likelihood over all samples in X
        """
        return self.score_samples(X).mean()
    

    def bic(self, X):
        """
        Bayesian information criterion for the current model over the input X

        Parameters:
        - X (matrix): Data

        Returns:
        - bic (float): BIC score
        """
        return -2 * self.score(X) * X.shape[0] + self._n_parameters() * np.log(X.shape[0])

    def aic(self, X):
        """Akaike information criterion for the current model over the input X

        Parameters:
        - X (matrix): Data

        Returns:
        - aic (float): AIC score
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()
        

    # def save_model(self, file_path):
    #     """
    #     Saved the model in h5 format

    #     Parameters:
    #     - file_path (string): Path to the file to create and save to
    #     """
    #     if self.converged == None:
    #         print("Model untrained, nothing to save")
    #         return
        
    #     # Saved as one list for simplicity
    #     meta_info = [self.n_observations, self.n_components, self.n_mixtures, self.converged, self.n_jobs]
    #     with h5py.File(file_path, "w") as f:
    #         f.create_dataset("params", data=self.params_)
    #         f.create_dataset("weights", data=self.weights_)
    #         f.create_dataset("size", data=meta_info)

    
    # def load_model(self, file_path):
    #     """
    #     Loads a previous model from its h5 file

    #     Parameters:
    #     - file_path (string): Path to the file to restore from
    #     """
    #     with h5py.File(file_path, "r") as f:
    #         self.params_ = f["params"][()]
    #         self.weights_ = f["weights"][()]
    #         meta_info = f["size"][()]

    #     self.n_observations = meta_info[0]
    #     self.n_components = meta_info[1]
    #     self.n_mixtures = meta_info[2]
    #     self.converged = bool(meta_info[3])
    #     self.n_jobs = meta_info[4]


    def __str__(self):
        return (
            f"Multivariate Beta Mixture Model w/ {self.n_mixtures} mixtures\n"
            f"Features per mixture: {self.n_components}\n"
            f"Trained on a {self.n_observations}x{self.n_components} data matrix\n"
            f"Converged: {self.converged}"
        )


class DirichletMM:
    def __init__(self, n_mixtures=1, verbose=False, verbose_interval=10, random_state=1):
        """
        Initializes the Dirichlet mixture model.

        Parameters:
        - n_mixtures (int): Number of Dirichlet distributions in the model.
        - verbose (boolean): If true, will print information during training.
        - verbose_interval (int): Number of iterations between verbose statements.
        - random_state (int): Random state for all algorithms.
        """
        self.n_mixtures = n_mixtures
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.random_state = random_state
        self.converged = False

    def _initialize(self, X, init_num):
        """
        Initializes the model parameters based on the given method.

        Parameters:
        - X (matrix): Data to initialize with.
        - init_num (int): Initialization number. Updates the random_state.
        """
        if self.method == "kmeans":
            resp = np.zeros(shape=(self.n_observations, self.n_mixtures))
            label = KMeans(n_clusters=self.n_mixtures, n_init=1, random_state=(self.random_state + init_num)).fit(X).labels_
            resp[np.arange(self.n_observations), label] = 1
        elif self.method == "random":
            np.random.seed(self.random_state + init_num)
            resp = np.random.uniform(size=(self.n_observations, self.n_mixtures))
            resp /= resp.sum(axis=1)[:, np.newaxis]
        
        resp += 10 * np.finfo(resp.dtype).eps

        self.params_ = np.random.rand(self.n_mixtures, self.n_components)
        self._m_step(X, np.log(resp))
        self._verbose_initialization(init_num)

    def _estimate_log_weights(self):
        """
        Computes the log weights of the current model.

        Returns:
        - log_weights (vector): Natural logarithm of the model weights.
        """
        return np.log(self.weights_)

    def _compute_log_prob_for_mixture(self, X, mix):
        """
        Helper function to compute the log probability for a single mixture.

        Parameters:
        - X (matrix): Data.
        - mix (int): Mixture number to assess the log probability of.

        Returns:
        - log_prob (vector): Log probabilities of each observation associated with this mixture.
        """
        alpha = self.params_[mix]

        log_dirichlet_const = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))
        log_prob = (log_dirichlet_const +
                    np.sum((alpha - 1) * np.log(X), axis=1))
        return log_prob

    def _estimate_log_prob(self, X):
        """
        Estimates the log probability for all the mixtures.

        Parameters:
        - X (matrix): Data.

        Returns:
        - log_prob (matrix): Matrix of log probabilities.
        """
        if self.n_jobs == 1:
            log_prob = np.empty((self.n_observations, self.n_mixtures))
            for mix in range(self.n_mixtures):
                log_prob[:, mix] = self._compute_log_prob_for_mixture(X, mix)
            return log_prob
        else:
            log_prob = Parallel(n_jobs=self.n_jobs)(
                delayed(self._compute_log_prob_for_mixture)(X, mix) for mix in range(self.n_mixtures)
            )
            return np.array(log_prob).T

    def _estimate_weighted_log_prob(self, X):
        """
        Estimates the weighted log probabilities for all the mixtures.

        Parameters:
        - X (matrix): Data.

        Returns:
        - weighted_log_prob (matrix): Matrix of weighted log probabilities.
        """
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    def _estimate_log_prob_resp(self, X):
        """
        Estimates the normalized log probabilities and the log responsibilities of each mixture.

        Parameters:
        - X (matrix): Data.

        Returns:
        - log_prob_norm (vector): Normalizing constant for each observation.
        - log_resp (matrix): Matrix of log responsibilities.
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)

        with np.errstate(under="ignore"):
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

        return log_prob_norm, log_resp

    def _e_step(self, X):
        """
        Performs the expectation step of the EM algorithm.

        Parameters:
        - X (matrix): Data.

        Returns:
        - mean_log_prob_norm (matrix): Mean normalizing constant for all the observations.
        - log_resp (matrix): Matrix of log responsibilities.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    def exp_responsibilities(self, log_resp):
        """
        Exponentiate the log responsibilities and compute the weighted importance of each mixture.

        Parameters:
        - log_resp (matrix): Matrix of log responsibilities.

        Returns:
        - resp (matrix): Exponentiated log responsibilities matrix.
        - nk (vector): Weighted importance of each mixture.
        """
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        return resp, nk

    def update_weights(self, nk):
        """
        Updates the weights of the mixtures.

        Parameters:
        - nk (vector): The sum of the probabilities of each mixture over every observation.
        """
        self.weights_ = nk / np.sum(nk)

    def _m_step(self, X, log_resp):
        """
        Performs the M step of the EM algorithm to update model parameters.

        Parameters:
        - X (matrix): Data.
        - log_resp (matrix): Matrix of log responsibilities.
        """
        resp, nk = self.exp_responsibilities(log_resp)
        self.update_weights(nk)

        for mix in range(self.n_mixtures):
            alpha_new = resp[:, mix] @ X
            self.params_[mix] = alpha_new / nk[mix]

    def _verbose_initialization(self, n):
        if self.verbose:
            print(f"New {self.method} initialization. Init Number {n + 1}")

    def _verbose_iter(self, iter, lower_bound, avg_time):
        if self.verbose and iter % self.verbose_interval == 0:
            print(f"Training Iteration {iter} complete. Current log prob lower bound: {lower_bound}. Avg Training Time: {np.round(avg_time, decimals=1)}s")

    def _verbose_converged(self, iter, lower_bound):
        if self.verbose:
            print(f"Converged on iteration {iter}. Log probability lower bound: {lower_bound}")

    def fit(self, X, n_init=3, method="kmeans", max_iter=200, tol=1e-4, n_jobs=8):
        """
        Fits the parameters and weights of the Dirichlet model to maximize the log-likelihood of the model
        given the data X.

        Parameters:
        - X (matrix): Data to fit the model to.
        - n_init (int): Number of initializations to try.
        - max_iter (int): Maximum number of iterations allowed if convergence is not yet reached.
        - tol (float): Minimum allowed difference in log likelihoods before convergence is reached.
        - n_jobs (int): Number of CPU cores to use on the E-Step.

        Returns:
        - self
        """
        self.n_observations, self.n_components = X.shape

        self.n_jobs = n_jobs
        self.converged = False
        max_lower_bound = -np.inf
        self.method = method if method.lower() in ["kmeans", "random"] else "kmeans"

        for init in range(n_init):
            print(f'{init + 1}-th DMM initialization')
            self._initialize(X, init)
            lower_bound = -np.inf

            start = time.time()
            for iter in tqdm(range(max_iter)):
                prev_lower_bound = lower_bound

                if iter % self.verbose_interval == 0:
                    start = time.time()
                    start_iter = iter

                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)

                lower_bound = log_prob_norm
                change = lower_bound - prev_lower_bound

                end = time.time()
                avg_time = (end - start) / (iter - start_iter + 1)

                self._verbose_iter(iter, lower_bound, avg_time)

                if abs(change) < tol:
                    self._verbose_converged(iter, lower_bound)
                    self.converged = True
                    break

            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound

                _, nk = self.exp_responsibilities(log_resp)
                self.update_weights(nk)

                best_params = [self.weights_, self.params_]

        self.weights_ = best_params[0]
        self.params_ = best_params[1]
    
    def predict_proba(self, X):
        """
        Predicts the probability that X belongs to each of the distributions.

        Parameters:
        - X (matrix): Data to probabilistically evaluate.

        Returns:
        - Probs (matrix): NxK matrix where K is number of mixtures. ij is the probability obs i belongs to mixture k.
        """
        _, log_resp = self._e_step(X)
        return np.exp(log_resp)
    
    def predict(self, X):
        """
        Predicts the most likely distribution for each observation in X.

        Parameters:
        - X (matrix): Data to predict classes of.

        Returns:
        - Classes (vector): The predicted classes [0, K-1] of the observations.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def score_samples(self, X):
        """
        Compute the log likelihood of each sample.

        Parameters:
        - X (matrix): Data.

        Returns:
        - log_prob: Log likelihood of each sample under the model.
        """
        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)
    
    def score(self, X):
        """
        Compute average log likelihood over all samples.

        Parameters:
        - X (matrix): Data.

        Returns:
        - avg_log_prob: Average log likelihood over all samples in X.
        """
        return self.score_samples(X).mean()
    
    def bic(self, X):
        """
        Bayesian information criterion for the current model over the input X.

        Parameters:
        - X (matrix): Data.

        Returns:
        - bic (float): BIC score.
        """
        return -2 * self.score(X) * X.shape[0] + self._n_parameters() * np.log(X.shape[0])

    def aic(self, X):
        """
        Akaike information criterion for the current model over the input X.

        Parameters:
        - X (matrix): Data.

        Returns:
        - aic (float): AIC score.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()
    
    def _n_parameters(self):
        """
        Returns the number of free parameters in the current model.

        Returns:
        - n_parameters (int): Number of free parameters in the model.
        """
        return int(self.n_components * self.n_mixtures - 1)

    def show_info(self):
        """
        Shows relevant information about the model. Includes the number of mixtures, training set info, and the status of model convergence.
        """
        print(f"Dirichlet Mixture Model\n"
              f"Number of mixtures: {self.n_mixtures}\n"
              f"Number of features: {self.n_components}\n"
              f"Converged: {self.converged}\n"
              f"Log-likelihood lower bound: {self.max_lower_bound}")
    
    def set_verbose(self, verbose, interval):
        """
        Update the verbose parameters.

        Parameters:
        - verbose (boolean): If true, prints updates every interval iters.
        - interval (int): Frequency of verbose statements.
        """
        self.verbose = verbose
        self.verbose_interval = interval

    def set_jobs(self, n_jobs):
        """
        Updates n_jobs.

        Parameters:
        - n_jobs (int): Number of CPUs to use in training.
        """
        self.n_jobs = n_jobs


if __name__ == '__main__':

    # Generating some synthetic data for testing
    np.random.seed(0)
    n_samples = 500
    n_features = 3
    n_components = 3

    # Generate random probability vectors (each row sums to 1)
    X = np.random.dirichlet([0.2, 0.5, 0.3], size=n_samples)

    # Initialize and fit the Dirichlet Mixture Model
    dmm = DirichletMM(n_mixtures=n_components, verbose=True, verbose_interval=5, random_state=42)
    dmm.fit(X, n_init=3, method="kmeans", max_iter=50, tol=1e-4, n_jobs=1)

    pdb.set_trace()

    # Predict the mixture component for each sample
    predictions = dmm.predict(X)
    print("Predicted mixture components:\n", predictions)

    # Predict the probability of each sample belonging to each mixture component
    probabilities = dmm.predict_proba(X)
    print("Predicted probabilities for each mixture:\n", probabilities)

    # Evaluate the log likelihood of the samples
    log_likelihood = dmm.score(X)
    print("Log likelihood of the data:", log_likelihood)

    # Calculate BIC and AIC scores
    # bic = dmm.bic(X)
    # aic = dmm.aic(X)
    # print(f"BIC: {bic}, AIC: {aic}")

import copy
import os
import pickle
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def compute_l1_norm(model1: nn.Module, model2: nn.Module) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes the L1 norm between the parameters of two models.

    Args:
        model1 (nn.Module): The first model.
        model2 (nn.Module): The second model.

    Returns:
        Tuple[torch.Tensor, Dict[str, float]]: A tuple containing the total L1 norm and a dictionary
        with the L1 norm for each layer.

    """
    norms = dict()
    l1_norm = 0.0
    for (n, p1), p2 in zip(model1.named_parameters(), model2.parameters()):
        layer_l1_norm = torch.norm(p1 - p2, 1)
        l1_norm += layer_l1_norm
        norms[n] = layer_l1_norm.item()

    return l1_norm, norms


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: List[int] = (1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def torch_load_old(save_path: str, device=None):
    with open(save_path, "rb") as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path, save_state_dict=True):
    # TODO: hacky way to save state dict
    if save_state_dict and isinstance(model, torch.nn.Module):
        model = model.state_dict()
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path, map_location="cpu", weights_only=False)
    if device is not None:
        model = model.to(device)
    return model


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def find_optimal_coef(
    results: Dict[str, Any],
    metric: str = "avg_normalized_top1",
    minimize: bool = False,
    control_metric: Optional[str] = None,
    control_metric_threshold: float = 0.0,
) -> float:
    """
    Finds the optimal coefficient based on the given results and metric.

    Args:
        results (Dict[str, Any]): A dictionary containing the results for different scaling coefficients.
        metric (str, optional): The metric to optimize. Defaults to "avg_normalized_top1".
        minimize (bool, optional): Whether to minimize the metric. Defaults to False.
        control_metric (str, optional): The control metric to check against. Defaults to None.
        control_metric_threshold (float, optional): The threshold value for the control metric. Defaults to 0.0.

    Returns:
        The optimal coefficient based on the given results and metric.
    """
    best_coef = None
    if minimize:
        best_metric = 1
    else:
        best_metric = 0
    for scaling_coef in results.keys():
        if control_metric is not None:
            if results[scaling_coef][control_metric] < control_metric_threshold:
                print(f"Control metric fell below {control_metric_threshold} threshold")
                continue
        if minimize:
            if results[scaling_coef][metric] < best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
        else:
            if results[scaling_coef][metric] > best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
    return best_coef


def find_optimal_coef_tradeoff(
    results: Dict[str, Any],
    tradeoff_target_weight: float = 5.0,
    minimize: bool = False,
    control_metric: Optional[str] = None,
    control_metric_threshold: float = 0.0,
) -> float:
    best_coef = None
    if minimize:
        best_metric = 1
    else:
        best_metric = 0
    for scaling_coef in results.keys():
        if minimize:
            if (
                tradeoff_target_weight * results[scaling_coef]["target_normalized_accuracy"]
                + results[scaling_coef]["control_normalized_accuracy"]
            ) < best_metric:
                best_metric = (
                    tradeoff_target_weight * results[scaling_coef]["target_normalized_accuracy"]
                    + results[scaling_coef]["control_normalized_accuracy"]
                )
                best_coef = scaling_coef
        else:
            if (
                tradeoff_target_weight * results[scaling_coef]["target_normalized_accuracy"]
                + results[scaling_coef]["control_normalized_accuracy"]
            ) > best_metric:
                best_metric = (
                    tradeoff_target_weight * results[scaling_coef]["target_normalized_accuracy"]
                    + results[scaling_coef]["control_normalized_accuracy"]
                )
                best_coef = scaling_coef
    return best_coef


def nonlinear_advantage(nonlinear_acc, linear_acc, num_classes):
    """Computes the normalized non-linear advantage of a finetuned model.

    The nonlinear_advantage is defined as:
        error_rate(linear_model) - error_rate(nonlinear_model) / (1 - 1 / num_classes)
    and takes values between [-1, 1]. A value of 0 indicates that the nonlinear
    model is no better than the linear one. Meanwhile, a value of 1 indicates
    that the nonlinear model is perfect and the linear trivial, and a value of
    -1 indicates the opposite.
    """
    return (nonlinear_acc - linear_acc) / (1.0 - 1.0 / num_classes)


def to_cuda(input_dict):
    cuda_dict = {}
    for key, value in input_dict.items():
        cuda_dict[key] = value.to("cuda")
    return cuda_dict


def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector([value.reshape(-1) for key, value in sorted_shared_state_dict.items()])


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict["transformer.shared.weight"]
    return sorted_reference_dict


def add_ptm_to_tv(tv_dict, ptm_dict):
    assert set(tv_dict.keys()) == set(ptm_dict.keys()), "Differing parameter names in models."
    final_dict = copy.deepcopy(tv_dict)
    for k, v in ptm_dict.items():
        final_dict[k] = tv_dict[k] + v
    return final_dict


def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )


def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True


def topk_values_mask(M, K=0.7, return_mask=False, reshape_mask=False):
    if K == 100:
        # print("Not applying mask")
        if return_mask:
            return M, torch.ones_like(M), None
        else:
            return M, torch.ones_like(M)

    if K >= 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if reshape_mask:
        final_mask = final_mask.reshape(M.shape)

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    else:
        return M * final_mask, final_mask.float().mean(dim=1)


def cleanup_linear(state_dict):
    # The linear model also has keys for the reference point $\theta_0$ in the state dict with the prefix `params0`.
    state_dict = {k: v for k, v in state_dict.items() if "params." in k}
    return state_dict


def get_ptm_linear(state_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # rename keys so that they match afterwards
    state_dict_new = {k.replace("params0", "params"): v for k, v in state_dict.items() if "params0." in k}
    state_dict_remaining = {k: v for k, v in state_dict.items() if "params." not in k}

    return state_dict_new, state_dict_remaining
