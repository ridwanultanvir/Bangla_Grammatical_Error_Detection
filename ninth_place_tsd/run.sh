#! /bin/bash
ln -s /mnt/y/_data/vasha23/results/ ./results

# bert_token
CUDA_VISIBLE_DEVICES=0 python -u train.py --train ./configs/bert_token/train.yaml --data ./configs/bert_token/dataset.yaml > out.log

CUDA_VISIBLE_DEVICES=0 python eval.py --eval ./configs/bert_token/eval.yaml


# bert_spans
CUDA_VISIBLE_DEVICES=0 python -u train.py --train ./configs/bert_spans/train.yaml --data ./configs/bert_spans/dataset.yaml > out.log

CUDA_VISIBLE_DEVICES=0 python eval.py --eval ./configs/bert_spans/eval.yaml

# bert_crf_token
CUDA_VISIBLE_DEVICES=0 python -u train.py --train ./configs/bert_crf_token/train.yaml --data ./configs/bert_crf_token/dataset.yaml > out.log
CUDA_VISIBLE_DEVICES=0 python -u eval.py --eval ./configs/bert_crf_token/eval.yaml

# Combine preds
python ./src/utils/combine_preds.py --config ./configs/combine_predictions/union_roberta_token_best_3_ckpts.yaml
