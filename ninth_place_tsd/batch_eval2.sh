#! /bin/bash

start=500
# start=3000
endin=10000
for (( COUNTER=start; COUNTER<=endin; COUNTER+=500 )); do
    echo checkpoint-$COUNTER
    CUDA_VISIBLE_DEVICES=0 python eval.py --eval ./configs/bert_token/eval2.yaml --model_checkpoint_name checkpoint-$COUNTER
done


