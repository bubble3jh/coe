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

top1_mod, top5_mod, n_mod = 0., 0., 0.
top5_modified_predictions = {classname: [] for classname in openai_classnames}  # To store top-5 predictions
model.eval()
zeroshot_weights = zeroshot_classifier(args, model, openai_classnames, openai_imagenet_template)

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Evaluating modified embeddings"):
        batch = maybe_dictionarize_batch(batch)
        images, labels = batch['images'].cuda(), batch['labels'].cuda()

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Use modified weights
        logits = 100. * image_features @ zeroshot_weights

        # Compute accuracies
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        top1_mod += acc1
        top5_mod += acc5
        n_mod += images.size(0)

# Compute final accuracy percentages
top1_mod = (top1_mod / n_mod) * 100
top5_mod = (top5_mod / n_mod) * 100


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

