import numpy as np
import torch
import clip
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import json
import os
import argparse
import random
from imagenet_classnames import openai_classnames
from adjustText import adjust_text
from sklearn.decomposition import PCA

# Configuration parser
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='ViT-B/32', help='CLIP model name')
    parser.add_argument("--dataset", default="ImageNet", help="Dataset name")
    parser.add_argument("--target_coef", type=float, default=1.0)
    parser.add_argument("--group_coef", type=float, default=1.0)
    parser.add_argument("--negative_coef", type=float, default=1.0)
    parser.add_argument("--ignore_wandb", action="store_true", default=False)
    return parser.parse_args()

# Embedding combination function
def combine_embeddings(embeddings, idx, target_coef, group_coef, negative_coef):
    combined = (target_coef * embeddings['target'][idx] +
                group_coef * embeddings['group'][idx] -
                negative_coef * embeddings['negative'][idx])
    return combined / combined.norm()

# Find group for a given class
def find_class_group(grouped_data, class_name):
    for group, classes in grouped_data.items():
        if class_name in classes:
            return group
    return None

# Visualization function for group analysis
def visualize_group_analysis(args, class_name, group_name, group_classes, classnames, embeddings, top5_predictions):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plt.rcParams.update({'font.size': 20, 'axes.titlesize': 20, 'axes.labelsize': 20})
    
    # Collect embeddings
    all_embeddings = []
    labels = []
    class_labels = []
    
    # Selected class components
    cls_idx = classnames.index(class_name)
    
    # Group center
    group_emb = torch.mean(torch.stack([embeddings['target'][classnames.index(c)] for c in group_classes]), dim=0)
    all_embeddings.append(group_emb.cpu().numpy())
    labels.append('group_center')
    class_labels.append(f'Group Center\n({group_name})')
    
    # Original embedding
    all_embeddings.append(embeddings['target'][cls_idx].cpu().numpy())
    labels.append('original')
    class_labels.append(f'Original\n{class_name}')
    
    # Combined embedding
    comb_emb = combine_embeddings(embeddings, cls_idx, 
                                args.target_coef, 
                                args.group_coef, 
                                args.negative_coef).cpu().numpy()
    all_embeddings.append(comb_emb)
    labels.append('combined')
    class_labels.append(f'Combined\n{class_name}')
    
    # Group members (max 10)
    for c in group_classes[:10]:
        if c == class_name: continue
        idx = classnames.index(c)
        all_embeddings.append(embeddings['target'][idx].cpu().numpy())
        labels.append('group_member')
        class_labels.append(c)
    
    # Hard negatives (from group visualization)
    negatives = top5_predictions.get(class_name, [])[:3]
    for neg in negatives:
        if neg in classnames:
            idx = classnames.index(neg)
            all_embeddings.append(embeddings['target'][idx].cpu().numpy())
            labels.append('negative')
            class_labels.append(neg)
    
    # t-SNE transformation
    n_samples = len(all_embeddings)
    valid_perplexity = min(10, n_samples-1) if n_samples > 1 else 1
    if valid_perplexity < 10:
        print(f"Warning: Adjusted perplexity to {valid_perplexity} for group analysis (n_samples={n_samples})")
    tsne = TSNE(n_components=2, perplexity=valid_perplexity, random_state=42)
    emb_2d = tsne.fit_transform(np.array(all_embeddings))
    
    # Visualization
    plot_common(emb_2d, labels, class_labels, 
                f"{class_name} Group Analysis ({group_name})", 
                f'./visualizations/{class_name}/group_analysis.png')

# Visualization function for top-5 analysis
def visualize_top5_analysis(args, class_name, classnames, embeddings, top5_predictions):
    plt.rcParams.update({'font.size': 20, 'axes.titlesize': 20, 'axes.labelsize': 20})
    
    # Collect embeddings
    all_embeddings = []
    labels = []
    class_labels = []
    
    cls_idx = classnames.index(class_name)
    
    # Original embedding
    all_embeddings.append(embeddings['target'][cls_idx].cpu().numpy())
    labels.append('original')
    class_labels.append(f'Original\n{class_name}')
    
    # Combined embedding
    comb_emb = combine_embeddings(embeddings, cls_idx, 
                                args.target_coef, 
                                args.group_coef, 
                                args.negative_coef).cpu().numpy()
    all_embeddings.append(comb_emb)
    labels.append('combined')
    class_labels.append(f'Combined\n{class_name}')
    
    # Top-5 hard negatives
    negatives = top5_predictions.get(class_name, [])[:5]
    for neg in negatives:
        if neg in classnames:
            idx = classnames.index(neg)
            all_embeddings.append(embeddings['target'][idx].cpu().numpy())
            labels.append('negative')
            class_labels.append(neg)
    
    # t-SNE transformation
    n_samples = len(all_embeddings)
    valid_perplexity = min(5, n_samples-1) if n_samples > 1 else 1
    if valid_perplexity < 5:
        print(f"Warning: Adjusted perplexity to {valid_perplexity} for top5 analysis (n_samples={n_samples})")
    tsne = TSNE(n_components=2, perplexity=valid_perplexity, random_state=42)
    emb_2d = tsne.fit_transform(np.array(all_embeddings))
    
    # Visualization
    plot_common(emb_2d, labels, class_labels, 
                f"{class_name} Top-5 Hard Negatives Analysis", 
                f'./visualizations/{class_name}/top5_analysis.png')

def validate_self_negative(args, class_name, classnames, embeddings, top5_predictions):
    """자가 클래스가 negative에 포함된 경우 임베딩 검증"""
    if class_name not in top5_predictions.get(class_name, []):
        return

    print("\n[자가 클래스 Negative 검증 시작]")
    
    # 기본 정보 추출
    cls_idx = classnames.index(class_name)
    self_neg_idx = classnames.index(class_name)  # 동일 클래스
    
    # 1. 임베딩 일치 여부 검사
    is_same_embedding = torch.allclose(
        embeddings['target'][cls_idx],
        embeddings['target'][self_neg_idx],
        atol=1e-6
    )
    print(f"- 임베딩 일치 여부: {is_same_embedding}")
    
    # 2. 코사인 유사도 계산
    cosine_sim = torch.nn.functional.cosine_similarity(
        embeddings['target'][cls_idx].unsqueeze(0),
        embeddings['target'][self_neg_idx].unsqueeze(0),
        dim=1
    ).item()
    print(f"- 코사인 유사도: {cosine_sim:.4f}")
    
    # 3. t-SNE 재현성 테스트
    test_points = np.stack([
        embeddings['target'][cls_idx].cpu().numpy(),
        embeddings['target'][self_neg_idx].cpu().numpy()
    ])
    
    # 다른 perplexity로 테스트
    for perplexity in [5, 15, 30]:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        emb_2d = tsne.fit_transform(test_points)
        distance = np.linalg.norm(emb_2d[0] - emb_2d[1])
        print(f"  Perplexity {perplexity}: 점간 거리 {distance:.2f}")
    
    # 4. PCA로 교차 검증
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(test_points)
    pca_distance = np.linalg.norm(pca_emb[0] - pca_emb[1])
    print(f"- PCA 거리: {pca_distance:.2f}")

def plot_common(emb_2d, labels, class_labels, title, filename):
    colors = {
        'original': '#4C72B0',
        'combined': '#DD8452',
        'group_center': '#55A868',
        'group_member': '#8C8C8C',
        'negative': '#C44E52'
    }
    
    markers = {
        'original': 'o',
        'combined': 's',
        'group_center': '*',
        'group_member': '^',
        'negative': 'X'
    }
    
    plt.figure(figsize=(16, 12))
    
    # 산점도 플롯
    for label in colors.keys():
        mask = np.array(labels) == label
        if np.sum(mask) > 0:
            plt.scatter(
                emb_2d[mask, 0], emb_2d[mask, 1],
                c=colors[label],
                s=300 if label == 'group_center' else 200,
                marker=markers[label],
                alpha=0.9,
                edgecolors='w',
                linewidth=2
            )
    
    # Original -> Combined 화살표 연결
    for i, (label, cls_label) in enumerate(zip(labels, class_labels)):
        if label == 'original':
            orig_idx = i
            # Combined는 항상 Original 바로 다음에 위치
            comb_idx = i + 1
            if comb_idx < len(labels) and labels[comb_idx] == 'combined':
                plt.annotate(
                    '', 
                    xy=emb_2d[comb_idx], 
                    xytext=emb_2d[orig_idx],
                    arrowprops=dict(
                        arrowstyle="->", 
                        color='gray',
                        lw=1.5,
                        linestyle='--',
                        alpha=0.6
                    )
                )
                # 거리 표시
                plt.text(
                    (emb_2d[orig_idx][0] + emb_2d[comb_idx][0])/2,
                    (emb_2d[orig_idx][1] + emb_2d[comb_idx][1])/2,
                    f'Δ={np.linalg.norm(emb_2d[orig_idx]-emb_2d[comb_idx]):.2f}',
                    color='gray',
                    fontsize=10,
                    alpha=0.8
                )
    
    # 기존 주석 및 텍스트 추가
    texts = []
    for i, (label, cls_label) in enumerate(zip(labels, class_labels)):
        if label != 'group_member':
            text = plt.text(
                emb_2d[i, 0], emb_2d[i, 1],
                cls_label,
                fontsize=20,
                ha='center',
                va='center',
                color=colors[label],
                weight='bold' if label == 'group_center' else 'normal'
            )
            texts.append(text)
    
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    
    # 범례 생성
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=colors['group_center'],
                  markersize=15, label='Group Center'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['original'],
                  markersize=10, label='Original Embeddings'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['combined'],
                  markersize=10, label='Combined Embeddings'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor=colors['negative'],
                  markersize=10, label='Hard Negatives')
    ]
    
    plt.legend(handles=legend_elements, loc='best', fontsize=20)
    plt.title(title, pad=20)
    plt.grid(True, alpha=0.2)
    plt.axis('off')
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {filename}")

def visualize_cross_group_analysis(args, class_name, original_group, other_group, group_classes, other_group_classes, classnames, embeddings):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plt.rcParams.update({'font.size': 20, 'axes.titlesize': 20, 'axes.labelsize': 20})
    
    # Collect embeddings
    all_embeddings = []
    labels = []
    class_labels = []
    
    # Selected class components
    cls_idx = classnames.index(class_name)
    
    # Original group center
    group_emb = torch.mean(torch.stack([embeddings['target'][classnames.index(c)] for c in group_classes]), dim=0)
    all_embeddings.append(group_emb.cpu().numpy())
    labels.append('group_center')
    class_labels.append(f'Group Center\n({original_group})')
    
    # Other group center
    other_group_emb = torch.mean(torch.stack([embeddings['target'][classnames.index(c)] for c in other_group_classes]), dim=0)
    all_embeddings.append(other_group_emb.cpu().numpy())
    labels.append('other_group_center')
    class_labels.append(f'Group Center\n({other_group})')
    
    # Original embedding
    all_embeddings.append(embeddings['target'][cls_idx].cpu().numpy())
    labels.append('original')
    class_labels.append(f'Original\n{class_name}')
    
    # Combined embedding
    comb_emb = combine_embeddings(embeddings, cls_idx, 
                                args.target_coef, 
                                args.group_coef, 
                                args.negative_coef).cpu().numpy()
    all_embeddings.append(comb_emb)
    labels.append('combined')
    class_labels.append(f'Combined\n{class_name}')
    
    # Group members (max 10)
    for c in group_classes[:10]:
        if c == class_name: continue
        idx = classnames.index(c)
        all_embeddings.append(embeddings['target'][idx].cpu().numpy())
        labels.append('group_member')
        class_labels.append(c)
    
    # Other group members (max 10)
    for c in other_group_classes[:10]:
        if c == class_name: continue
        idx = classnames.index(c)
        all_embeddings.append(embeddings['target'][idx].cpu().numpy())
        labels.append('other_group_member')
        class_labels.append(c)
    
    # Hard negatives (from group visualization)
    negatives = top5_predictions.get(class_name, [])[:3]
    for neg in negatives:
        if neg in classnames:
            idx = classnames.index(neg)
            all_embeddings.append(embeddings['target'][idx].cpu().numpy())
            labels.append('negative')
            class_labels.append(neg)
    
    # t-SNE transformation
    n_samples = len(all_embeddings)
    valid_perplexity = min(15, n_samples-1) if n_samples > 1 else 1
    if valid_perplexity < 15:
        print(f"Warning: Adjusted perplexity to {valid_perplexity} for cross-group analysis (n_samples={n_samples})")
    tsne = TSNE(n_components=2, perplexity=valid_perplexity, random_state=42)
    emb_2d = tsne.fit_transform(np.array(all_embeddings))
    
    # Visualization
    plot_cross_group(emb_2d, labels, class_labels, 
                    f"{class_name} Cross-Group Analysis ({original_group} vs {other_group})", 
                    f'./visualizations/{class_name}/cross_group_analysis.png')

def plot_cross_group(emb_2d, labels, class_labels, title, filename):
    colors = {
        'original': '#4C72B0',
        'combined': '#DD8452',
        'group_center': '#55A868',
        'other_group_center': '#CC79A7',
        'group_member': '#8C8C8C',
        'other_group_member': '#D55E00',
        'negative': '#C44E52'
    }

    markers = {
        'original': 'o',
        'combined': 's',
        'group_center': '*',
        'other_group_center': 'P',
        'group_member': '^',
        'other_group_member': 'v',
        'negative': 'X'
    }

    plt.figure(figsize=(20, 16))
    
    # 산점도 플롯
    for label in colors.keys():
        mask = np.array(labels) == label
        if np.sum(mask) > 0:
            plt.scatter(
                emb_2d[mask, 0], emb_2d[mask, 1],
                c=colors[label],
                s=300 if label == 'group_center' or label == 'other_group_center' else 200,
                marker=markers[label],
                alpha=0.9,
                edgecolors='w',
                linewidth=2
            )
    
    # Original -> Combined 화살표 연결
    for i, (label, cls_label) in enumerate(zip(labels, class_labels)):
        if label == 'original':
            orig_idx = i
            # Combined는 항상 Original 바로 다음에 위치
            comb_idx = i + 1
            if comb_idx < len(labels) and labels[comb_idx] == 'combined':
                plt.annotate(
                    '', 
                    xy=emb_2d[comb_idx], 
                    xytext=emb_2d[orig_idx],
                    arrowprops=dict(
                        arrowstyle="->", 
                        color='gray',
                        lw=1.5,
                        linestyle='--',
                        alpha=0.6
                    )
                )
                # 거리 표시
                plt.text(
                    (emb_2d[orig_idx][0] + emb_2d[comb_idx][0])/2,
                    (emb_2d[orig_idx][1] + emb_2d[comb_idx][1])/2,
                    f'Δ={np.linalg.norm(emb_2d[orig_idx]-emb_2d[comb_idx]):.2f}',
                    color='gray',
                    fontsize=10,
                    alpha=0.8
                )
    
    # 기존 주석 및 텍스트 추가
    texts = []
    for i, (label, cls_label) in enumerate(zip(labels, class_labels)):
        if label != 'group_member' and label != 'other_group_member':
            text = plt.text(
                emb_2d[i, 0], emb_2d[i, 1],
                cls_label,
                fontsize=20,
                ha='center',
                va='center',
                color=colors[label],
                weight='bold' if label == 'group_center' or label == 'other_group_center' else 'normal'
            )
            texts.append(text)
    
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    
    # 범례 수정 - mpatches.Patch 대신 plt.Line2D 사용
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=colors['group_center'],
                  markersize=15, label='Original Group Center'),
        plt.Line2D([0], [0], marker='P', color='w', markerfacecolor=colors['other_group_center'],
                  markersize=15, label='Other Group Center'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['original'],
                  markersize=10, label='Original Embedding'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['combined'],
                  markersize=10, label='Combined Embedding'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=colors['group_member'],
                  markersize=10, label='Original Group Members'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=colors['other_group_member'],
                  markersize=10, label='Other Group Members')
    ]
    
    plt.legend(handles=legend_elements, loc='best', fontsize=14)
    plt.title(title, pad=25, fontsize=18)
    plt.grid(True, alpha=0.2)
    plt.axis('off')
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-group visualization: {filename}")

if __name__ == "__main__":
    args = parse_arguments()
    
    # Load data
    with open('./results/grouped_imagenet_classes.json') as f:
        grouped_data = json.load(f)
    
    classnames = openai_classnames
    
    # Load embeddings
    embed_path = f'./checkpoints/{args.dataset}_embeddings_{args.model.replace("/", "_").replace("-", "_")}.pt'
    embeddings = torch.load(embed_path, map_location='cpu')
    
    # Load top-5 predictions
    top5_path = f'./results/{args.dataset}_top5_predictions_{args.model.replace("/", "_").replace("-", "_")}.json'
    with open(top5_path) as f:
        top5_predictions = json.load(f)
    
    # Random class selection
    valid_classes = [c for c in classnames 
                    if find_class_group(grouped_data, c) 
                    and len(top5_predictions.get(c, [])) >= 5]
    selected_class = random.choice(valid_classes)
    group_name = find_class_group(grouped_data, selected_class)
    group_classes = [c for c in grouped_data[group_name] if c in classnames]
    
    print(f"Selected class: {selected_class}")
    print(f"Group: {group_name} ({len(group_classes)} classes)")
    print(f"Top-5 negatives: {top5_predictions[selected_class][:5]}")
    
    # Create visualizations
    os.makedirs('./visualizations', exist_ok=True)
    os.makedirs(f'./visualizations/{selected_class}', exist_ok=True)
    
    # Group analysis visualization
    visualize_group_analysis(args, selected_class, group_name, 
                            group_classes, classnames, embeddings, top5_predictions)
    
    # Top-5 analysis visualization
    visualize_top5_analysis(args, selected_class, classnames, 
                           embeddings, top5_predictions)
    
    # 자가 클래스 검증 추가
    # validate_self_negative(args, selected_class, classnames, embeddings, top5_predictions)
    
    # 다른 그룹 랜덤 선택
    other_groups = [g for g in grouped_data.keys() if g != group_name]
    other_group = random.choice(other_groups)
    other_group_classes = [c for c in grouped_data[other_group] if c in classnames]
    
    # 3가지 시각화 실행
    visualize_group_analysis(args, selected_class, group_name, 
                            group_classes, classnames, embeddings, top5_predictions)
    
    visualize_top5_analysis(args, selected_class, classnames, 
                           embeddings, top5_predictions)
    
    visualize_cross_group_analysis(args, selected_class, group_name, other_group,
                                  group_classes, other_group_classes,
                                  classnames, embeddings)
    
    # W&B 로깅 업데이트
    if not args.ignore_wandb:
        import wandb
        wandb.init(project="COE-Visualization", config=vars(args))
        wandb.log({
            "Group Analysis": wandb.Image(f'./visualizations/{selected_class}/group_analysis.png'),
            "Top5 Analysis": wandb.Image(f'./visualizations/{selected_class}/top5_analysis.png')
        })