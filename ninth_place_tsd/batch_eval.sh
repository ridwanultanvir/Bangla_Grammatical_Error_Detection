#! /bin/bash

start=500
# start=3000
endin=20000
# GPU=0
GPU=1
for (( COUNTER=start; COUNTER<=endin; COUNTER+=500 )); do
    echo checkpoint-$COUNTER
    # CUDA_VISIBLE_DEVICES=$GPU python eval.py --eval ./configs/bert_token/eval.yaml --model_checkpoint_name checkpoint-$COUNTER
    # CUDA_VISIBLE_DEVICES=$GPU python eval.py --eval ./configs/bert_token_3cls/eval.yaml --model_checkpoint_name checkpoint-$COUNTER
    # CUDA_VISIBLE_DEVICES=$GPU python eval.py --eval ./configs/bert_crf_token/eval.yaml --model_checkpoint_name checkpoint-$COUNTER
    CUDA_VISIBLE_DEVICES=$GPU python eval.py --eval ./configs/bert_crf_3cls_token/eval.yaml --model_checkpoint_name checkpoint-$COUNTER
    # CUDA_VISIBLE_DEVICES=$GPU python eval.py --eval ./configs/bert_spans/eval.yaml --model_checkpoint_name checkpoint-$COUNTER
done


