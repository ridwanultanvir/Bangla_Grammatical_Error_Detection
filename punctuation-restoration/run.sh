#! /bin/bash

python -u src/train.py --cuda=True \
  --pretrained-model=xlm-roberta-large --freeze-bert=False --lstm-dim=-1 \
  --language=bangla --seed=1 --lr=5e-6 --epoch=10 --use-crf=False --augment-type=all  --augment-rate=0.15 \
  --alpha-sub=0.4 --alpha-del=0.4 --data-path=data --save-path=out


# Train with all augmentations
python -u src/train.py --cuda=True \
  --pretrained-model=xlm-roberta-large --freeze-bert=False --lstm-dim=-1 \
  --language=bangla --seed=1 --lr=5e-6 --epoch=10 --use-crf=False --augment-type=all  --augment-rate=0.15 \
  --alpha-sub=0.4 --alpha-del=0.4 --data-path=data --save-path=out

# Train with no augmentations
python -u src/train.py --cuda=True \
  --pretrained-model=xlm-roberta-large --freeze-bert=False --lstm-dim=-1 \
  --language=bangla --seed=1 --lr=5e-6 --epoch=100 --use-crf=False --augment-type=none  --augment-rate=0.15 \
  --alpha-sub=0.4 --alpha-del=0.4 --data-path=data --save-path=out

# Normal inference for input
python -u src/inference.py --pretrained-model=xlm-roberta-large \
  --weight-path=out/weights.pt --language=bn  \
  --in-file=data/test_bn.txt --out-file=data/test_bn_out.txt

# Normal inference for given input
python -u src/inference.py --pretrained-model=xlm-roberta-large \
  --weight-path=out/weights.pt --language=bn  \
  --in-file=data/test_bn2.txt --out-file=data/test_bn_out2.txt

# GED inference xlmrl_gedtrain_ep10
CUDA_VISIBLE_DEVICES=1 python -u src/inference_ged.py --pretrained-model=xlm-roberta-large \
  --weight-path=out/xlmrl_gedtrain_ep10/weights.pt --language=bn  \
  --in-file=data/test_bn2.txt --out-file=data/test_bn_out2.txt

# GED inference
python -u src/inference_ged.py --pretrained-model=xlm-roberta-large \
  --weight-path=out/weights.pt --language=bn  \
  --in-file=data/test_bn2.txt --out-file=data/test_bn_out2.txt

  
