#! /bin/bash
ln -s /mnt/y/_data/vasha23/results/ ./results

cd ninth_place_tsd

# bert_token
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --train ./configs/bert_token/train.yaml --data ./configs/bert_token/dataset.yaml

CUDA_VISIBLE_DEVICES=0 python eval.py --eval ./configs/bert_token/eval.yaml

# bert_token for 3 classes
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --train ./configs/bert_token_3cls/train.yaml --data ./configs/bert_token_3cls/dataset.yaml

CUDA_VISIBLE_DEVICES=0 python eval.py --eval ./configs/bert_token_3cls/eval.yaml

# bert_token for 4 classes v2
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --train ./configs/bert_token_4cls_v2/train.yaml --data ./configs/bert_token_4cls_v2/dataset.yaml

CUDA_VISIBLE_DEVICES=0 python eval.py --eval ./configs/bert_token_4cls_v2/eval.yaml

# bert_token for 4 classes
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --train ./configs/bert_token_4cls/train.yaml --data ./configs/bert_token_4cls/dataset.yaml

CUDA_VISIBLE_DEVICES=0 python eval.py --eval ./configs/bert_token_4cls/eval.yaml


# bert_crf_4cls_token
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --train ./configs/bert_crf_4cls_token/train.yaml --data ./configs/bert_crf_4cls_token/dataset.yaml

CUDA_VISIBLE_DEVICES=0 python -u eval.py --eval ./configs/bert_crf_4cls_token/eval.yaml


# bert_spans
CUDA_VISIBLE_DEVICES=0 python -u train.py --train ./configs/bert_spans/train.yaml --data ./configs/bert_spans/dataset.yaml

CUDA_VISIBLE_DEVICES=0 python eval.py --eval ./configs/bert_spans/eval.yaml

# bert_crf_token
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --train ./configs/bert_crf_token/train.yaml --data ./configs/bert_crf_token/dataset.yaml

CUDA_VISIBLE_DEVICES=0 python -u eval.py --eval ./configs/bert_crf_token/eval.yaml

# bert_crf_3cls_token
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --train ./configs/bert_crf_3cls_token/train.yaml --data ./configs/bert_crf_3cls_token/dataset.yaml

CUDA_VISIBLE_DEVICES=0 python -u eval.py --eval ./configs/bert_crf_3cls_token/eval.yaml

# Combine preds
python ./src/utils/combine_preds.py --config ./configs/combine_predictions/union_roberta_token_best_3_ckpts.yaml

python ./src/utils/combine_preds_3cls.py --config ./configs/combine_predictions/union_roberta_token_best_3_ckpts.yaml

python ./src/utils/combine_preds_4cls.py --config ./configs/combine_predictions/union_roberta_token_best_3_ckpts.yaml
