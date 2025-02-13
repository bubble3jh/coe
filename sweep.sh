#!/bin/bash

# 모델 리스트
models=("ViT-B/32" "ViT-L/14")

# coefficient 조합 리스트
target_coefs=(0.5 0.75 1.0 1.25 1.5)
group_coefs=(0.0 0.5 0.75 1.0 1.25 1.5)
negative_coefs=(0.0 0.5 0.75 1.0 1.25 1.5)

# 데이터셋 고정
dataset="ImageNet"

# 실험 실행
for model in "${models[@]}"; do
    for target_coef in "${target_coefs[@]}"; do
        for group_coef in "${group_coefs[@]}"; do
            for negative_coef in "${negative_coefs[@]}"; do
                echo "Running experiment with model: $model, target_coef: $target_coef, group_coef: $group_coef, negative_coef: $negative_coef"
                python zs_jh.py \
                    --model "$model" \
                    --dataset "$dataset" \
                    --target_coef "$target_coef" \
                    --group_coef "$group_coef" \
                    --negative_coef "$negative_coef"
            done
        done
    done
done