import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets

import clip
from openai_imagenet_template import openai_imagenet_template
# from imagenet_classnames import imagenet_classnames  # (따로 준비되어 있다고 가정)

device = "cuda" if torch.cuda.is_available() else "cpu"

###############################################
# 1. Load CLIP model (ViT-B/32)
###############################################
model_name = "ViT-B/32"  
model, preprocess = clip.load(model_name, device=device, jit=False)
model.eval()

###############################################
# 2. Prepare your dataset & DataLoader
###############################################
# 예: ImageNet val 셋 경로를 지정 (또는 원하는 Dataset)
# 아래는 torchvision ImageNet 예시
imagenet_val_dir = "/path/to/imagenet/val"  # 실제 경로로 교체
batch_size = 64

val_dataset = datasets.ImageFolder(
    root=imagenet_val_dir,
    transform=preprocess
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4
)

# 클래스 이름: ImageFolder -> class_to_idx 순서대로
imagenet_classnames = list(val_dataset.class_to_idx.keys())
# 예시로, idx -> classname 순서가 val_dataset.classes와 동일합니다.

###############################################
# 3. Build zero-shot classifier 
#    using openai_imagenet_template
###############################################
def build_zeroshot_weights(classnames, templates, model):
    """
    여러 Template을 이용해 각 클래스별 텍스트 임베딩을 만든 뒤 평균을 취한다.
    결과 shape: (num_classes, D)
    """
    zeroshot_weights = []

    with torch.no_grad():
        for classname in classnames:
            texts = [template(classname) for template in templates]
            # tokenize: CLIP의 tokenizer 사용
            tokenized = clip.tokenize(texts).to(device)  # shape: (num_templates, max_len)
            # model.encode_text -> (num_templates, D)
            text_embeddings = model.encode_text(tokenized)
            # L2 normalize
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            # 여러 template 결과를 평균
            mean_text_embedding = text_embeddings.mean(dim=0)
            # 최종 L2 normalize
            mean_text_embedding = mean_text_embedding / mean_text_embedding.norm()
            zeroshot_weights.append(mean_text_embedding)

    # (num_classes, D) 텐서로 쌓기
    zeroshot_weights = torch.stack(zeroshot_weights, dim=0)
    return zeroshot_weights


###############################################
# 4. Compute the zeroshot classifier weights
###############################################
zeroshot_weights = build_zeroshot_weights(
    classnames=imagenet_classnames,
    templates=openai_imagenet_template,
    model=model
)
# shape: (num_classes, D)
num_classes = zeroshot_weights.shape[0]

###############################################
# 5. Inference loop (zero-shot classification)
###############################################
correct_top1 = 0
correct_top5 = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size_curr = images.size(0)

        # 1) 이미지 임베딩 추출
        image_features = model.encode_image(images)
        # 2) normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 3) 유사도 계산
        #    (batch_size, D) @ (D, num_classes) -> (batch_size, num_classes)
        logits = image_features @ zeroshot_weights.t()
        
        # 4) top-k accuracy 계산
        # top-1
        preds_top1 = logits.argmax(dim=1)
        correct_top1 += torch.sum(preds_top1 == labels).item()

        # top-5
        preds_top5 = logits.topk(5, dim=1).indices  # shape: (batch_size, 5)
        for i in range(batch_size_curr):
            if labels[i] in preds_top5[i]:
                correct_top5 += 1
        
        total += batch_size_curr

# 정확도
top1_acc = correct_top1 / total
top5_acc = correct_top5 / total
print(f"Zero-shot Top-1 accuracy: {top1_acc*100:.2f}%")
print(f"Zero-shot Top-5 accuracy: {top5_acc*100:.2f}%")
