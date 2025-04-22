import torch
from transformers import AutoProcessor, AutoModel
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 사용할 디바이스 설정 (GPU가 있다면 GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 테스트 데이터셋 로드 (PIL 이미지 형태, resize 적용)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 모델 입력 사이즈에 맞게 조정 (예: 224x224)
])
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# CIFAR-10 클래스 이름
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

# gemma 3.0 4b 모델과 프로세서 불러오기 (모델 이름은 실제 이름에 맞게 수정)
model_name = "gemma-3.0-4b"  # 실제 모델 ID로 변경 필요
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to(device)
model.eval()

import torch.nn.functional as F

def classify_image(image):
    """
    단일 이미지(PIL Image)에 대해 CIFAR-10 클래스별 텍스트 프롬프트를 이용해 zero-shot 분류 수행
    """
    # 각 클래스에 대해 프롬프트 생성
    texts = [f"a photo of a {label}" for label in cifar10_classes]
    # 모델 입력 생성 (이미지와 텍스트 함께 처리)
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    # 각 입력 텐서를 device로 이동
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # gemma 모델은 CLIP과 유사하게 image_embeds와 text_embeds를 출력한다고 가정합니다.
        image_embeds = outputs.image_embeds   # shape: [1, dim]
        text_embeds = outputs.text_embeds     # shape: [num_texts, dim]
    
    # 정규화
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # 코사인 유사도 계산 (각 텍스트와 이미지 간의 유사도)
    similarities = (image_embeds @ text_embeds.T).squeeze(0)  # shape: [num_texts]
    pred_index = similarities.argmax().item()
    return cifar10_classes[pred_index], similarities.cpu().numpy()

# CIFAR-10 테스트셋에 대해 평가 진행
correct = 0
total = 0
for img, label in tqdm(test_loader, desc="Evaluating"):
    # img는 transform을 거친 PIL 이미지임 (DataLoader로 불러온 경우 batch 차원 존재)
    predicted_label, _ = classify_image(img[0])
    true_label = cifar10_classes[label.item()]
    if predicted_label == true_label:
        correct += 1
    total += 1

accuracy = correct / total * 100
print(f"Accuracy on CIFAR-10 test set: {accuracy:.2f}%")
