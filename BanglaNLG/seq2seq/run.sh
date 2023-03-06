#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python ./run_seq2seq.py \
    --model_name_or_path "csebuetnlp/banglat5" --dataset_dir "banglaged/" --output_dir "outputs/" \
    --learning_rate=5e-4 \
    --warmup_steps 5000 \
    --label_smoothing_factor 0.1 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.1 \
    --lr_scheduler_type "linear"  \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --max_source_length 256 --max_target_length 256 \
    --logging_strategy "epoch" --save_strategy "epoch" --evaluation_strategy "epoch" \
    --source_key sentence --target_key gt --source_lang bn --target_lang bn \
    --greater_is_better true --load_best_model_at_end \
    --metric_for_best_model sacrebleu --evaluation_metric sacrebleu \
    --num_train_epochs 20 \
    --do_train --do_eval \
    --do_predict  --predict_with_generate \
    --resume_from_checkpoint "outputs/checkpoint-2340"

# --do_eval 
# --do_predict  --predict_with_generate