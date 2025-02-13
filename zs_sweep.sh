#!/bin/bash

# 직접 각 조합을 실행하도록 명시적으로 작성
echo "Running with alpha=0.3 and beta=0.1 on GPU 0"
CUDA_VISIBLE_DEVICES=0 python zs_mod.py --alpha 0.1 --beta 1.0 &

echo "Running with alpha=0.3 and beta=0.3 on GPU 1"
CUDA_VISIBLE_DEVICES=1 python zs_mod.py --alpha 0.1 --beta 1.2 &

echo "Running with alpha=0.3 and beta=0.5 on GPU 2"
CUDA_VISIBLE_DEVICES=2 python zs_mod.py --alpha 0.1 --beta 0.9 &

echo "Running with alpha=0.5 and beta=0.1 on GPU 3"
CUDA_VISIBLE_DEVICES=3 python zs_mod.py --alpha 0.3 --beta 0.9 &

echo "Running with alpha=0.5 and beta=0.3 on GPU 4"
CUDA_VISIBLE_DEVICES=4 python zs_mod.py --alpha 0.5 --beta 1.2 &

echo "Running with alpha=0.5 and beta=0.5 on GPU 5"
CUDA_VISIBLE_DEVICES=5 python zs_mod.py --alpha 0.5 --beta 0.9 &

echo "Running with alpha=0.7 and beta=0.1 on GPU 6"
CUDA_VISIBLE_DEVICES=6 python zs_mod.py --alpha 0.3 --beta 1.2 &

echo "Running with alpha=0.7 and beta=0.3 on GPU 7"
CUDA_VISIBLE_DEVICES=7 python zs_mod.py --alpha 0.7 --beta 0.9 &

# 모든 작업이 끝날 때까지 대기
wait
