#! /bin/bash

start=500
start=5000
# start=24000
# start=15500
endin=60000
# GPU=0
GPU=1
for (( COUNTER=start; COUNTER<=endin; COUNTER+=500 )); do
    echo checkpoint-$COUNTER
    # CUDA_VISIBLE_DEVICES=$GPU python eval.py --eval ./configs/bert_token/eval.yaml --model_checkpoint_name checkpoint-$COUNTER
    # CUDA_VISIBLE_DEVICES=$GPU python eval.py --eval ./configs/bert_token_3cls/eval.yaml --model_checkpoint_name checkpoint-$COUNTER
    
    CUDA_VISIBLE_DEVICES=$GPU python eval.py --eval ./configs/bert_token_4cls/eval.yaml --model_checkpoint_name checkpoint-$COUNTER

    # CUDA_VISIBLE_DEVICES=$GPU python eval.py --eval ./configs/bert_token_4cls_v2/eval.yaml --model_checkpoint_name checkpoint-$COUNTER
    # CUDA_VISIBLE_DEVICES=$GPU python eval.py --eval ./configs/bert_crf_token/eval.yaml --model_checkpoint_name checkpoint-$COUNTER
    # CUDA_VISIBLE_DEVICES=$GPU python eval.py --eval ./configs/bert_crf_3cls_token/eval.yaml --model_checkpoint_name checkpoint-$COUNTER
    # CUDA_VISIBLE_DEVICES=$GPU python eval.py --eval ./configs/bert_crf_4cls_token/eval.yaml --model_checkpoint_name checkpoint-$COUNTER
    # CUDA_VISIBLE_DEVICES=$GPU python eval.py --eval ./configs/bert_spans/eval.yaml --model_checkpoint_name checkpoint-$COUNTER
    # Sleep for 60 seconds
    # sleep 60
done


