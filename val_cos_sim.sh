device_ids=(0 1 2 3 4 5)
num_devices=${#device_ids[@]}
i=0
job_count=0

for beta in 1.5 1.2 1.0 0.8 0.5 0.2; do
    for method in text_svd_proj; do
        for custom_template in openai; do
        for mean_center in "" "--mean_center"; do
        for pc_mode in upper lower; do
        for top_l in 1 2 3 5 10; do
            gpu_id=${device_ids[$((i % num_devices))]}
            CUDA_VISIBLE_DEVICES=$gpu_id python validate_cos_sim.py --method $method --beta $beta --use_wandb --custom_template $custom_template $mean_center --pc_mode $pc_mode --top_l $top_l &

            i=$((i + 1))
            job_count=$((job_count + 1))

            # 동시에 6개 실행 후 기다림
                if (( job_count % num_devices == 0 )); then
                    wait
                fi
            done
            done
            done
        done
    done
done

# 남은 job들 기다리기
wait
