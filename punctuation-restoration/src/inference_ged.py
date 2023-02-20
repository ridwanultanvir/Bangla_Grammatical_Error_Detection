import re
import torch

import argparse
from model import DeepPunctuation, DeepPunctuationCRF
from config import *

parser = argparse.ArgumentParser(description='Punctuation restoration inference on text file')
parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
parser.add_argument('--pretrained-model', default='xlm-roberta-large', type=str, help='pretrained language model')
parser.add_argument('--lstm-dim', default=-1, type=int,
                    help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')
parser.add_argument('--use-crf', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to use CRF layer or not')
parser.add_argument('--language', default='en', type=str, help='language English (en) oe Bangla (bn)')
parser.add_argument('--in-file', default='data/test_en.txt', type=str, help='path to inference file')
parser.add_argument('--weight-path', default='xlm-roberta-large.pt', type=str, help='model weight path')
parser.add_argument('--sequence-length', default=256, type=int,
                    help='sequence length to use when preparing dataset (default 256)')
parser.add_argument('--out-file', default='data/test_en_out.txt', type=str, help='output file location')

args = parser.parse_args()

# tokenizer
tokenizer = MODELS[args.pretrained_model][1].from_pretrained(args.pretrained_model)
token_style = MODELS[args.pretrained_model][3]

# logs
model_save_path = args.weight_path

# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
if args.use_crf:
    deep_punctuation = DeepPunctuationCRF(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
else:
    deep_punctuation = DeepPunctuation(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
deep_punctuation.to(device)

def find_nth_occurrence(text, word, n):
  start = -1
  for i in range(n):
      start = text.find(word, start + 1)
      if start == -1:
          return -1
  return start


def inference():
    deep_punctuation.load_state_dict(torch.load(model_save_path))
    deep_punctuation.eval()

    thresh = 0.9
    # thresh = 0.8
    # thresh = 0.75
    # thresh = 0.7
    # thresh = 0.6
    # thresh = 0.5
    # thresh = 0.1
    suffix = ""
    suffix += "_" + f"{thresh: .2f}"
    suffix += "_ep46"

    # predfile = "preds/test_results_bn3500_bnlargefull1_int.csv"
    predfile = "../data/pred/3_cls/bert_large/spans-pred_test_checkpoint-6000_spell.csv"
    from pathlib import Path
    pred_path = Path(predfile)
    debug_file = pred_path.parent/(pred_path.stem + "_debug" + suffix + pred_path.suffix)
    submit_file = pred_path.parent/(pred_path.stem + "_submit" + suffix + pred_path.suffix)
    incol = "Expected"
    # predfile = "preds/DataSetFold1_u.csv"
    # incol = "gt"
    import pandas as pd
    from tqdm import tqdm
    import copy
    from collections import defaultdict
    preddf = pd.read_csv(predfile)
    # Iterate over the column incol
    output_expected = []
    
    p_file = open("preds/proba.log", "w")
    
    for index, row in tqdm(preddf.iterrows(), total=len(preddf)):
        # print(row[incol])
        text = row[incol]
        frequency = defaultdict(int)

        # with open(args.in_file, 'r', encoding='utf-8') as f:
        #     text = f.read()
        # text = re.sub(r"[,:\-–.!;?]", '', text) #! Why is not dari removed?
        # text = re.sub(r"[,:\-–.!;?]।", '', text) #! Why is not dari removed?

        original_text = copy.deepcopy(text)
        space_chars = ['।', '?', '!', ",", '$']

        text = re.sub(r"[,:\-–.!;?]।", '', text) #! Why is not dari removed?
        text = text.replace('$', '')
        words_original_case = text.split()
        words = text.lower().split()
        # print("Words: ", words_original_case)

        word_pos = 0
        sequence_len = args.sequence_length
        result = ""
        decode_idx = 0
        punctuation_map = {0: '', 1: ',', 2: '.', 3: '?'}
        if args.language != 'en':
            punctuation_map[2] = '।'
        
        changes = []

        while word_pos < len(words):
            x = [TOKEN_IDX[token_style]['START_SEQ']]
            y_mask = [0]

            while len(x) < sequence_len and word_pos < len(words):
                tokens = tokenizer.tokenize(words[word_pos])
                if len(tokens) + len(x) >= sequence_len:
                    break
                else:
                    for i in range(len(tokens) - 1):
                        x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                        y_mask.append(0)
                    x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                    y_mask.append(1)
                    word_pos += 1
            x.append(TOKEN_IDX[token_style]['END_SEQ'])
            y_mask.append(0)
            if len(x) < sequence_len:
                x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
                y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
            attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]

            x = torch.tensor(x).reshape(1,-1)
            y_mask = torch.tensor(y_mask)
            attn_mask = torch.tensor(attn_mask).reshape(1,-1)
            x, attn_mask, y_mask = x.to(device), attn_mask.to(device), y_mask.to(device)

            with torch.no_grad():
                if args.use_crf:
                    y = torch.zeros(x.shape[0])
                    y_predict = deep_punctuation(x, attn_mask, y)
                    y_predict = y_predict.view(-1)
                else:
                    y_predict = deep_punctuation(x, attn_mask)
                    y_predict = y_predict.view(-1, y_predict.shape[2])
                    y_proba = torch.softmax(y_predict, dim=1)
                    y_predict = torch.argmax(y_predict, dim=1).view(-1)
            for i in range(y_mask.shape[0]):
                if y_mask[i] == 1:
                    pred = y_predict[i].item()
                    result += words_original_case[decode_idx] + punctuation_map[pred] + ' '
                    decode_idx += 1
                    token = words_original_case[decode_idx-1]
                    # Find the nth occurence of the word in the original text
                    frequency[token] += 1

                    if pred == 0: continue
                    # Print to file
                    proba = y_proba[i].tolist()
                    proba = [str(f"{p:.4f}") for p in proba]
                    p_file.write(
                        # words_original_case[decode_idx-1] + " " + 
                        # punctuation_map[pred] + " " + 

                        result + "\n" +                        
                        str(proba)
                    )
                    p_file.write("\n")

                    if y_proba[i][pred] < thresh: 
                        continue
                    
                    orig_idx = find_nth_occurrence(original_text, token, frequency[token])

                    # orig_idx = original_text.find(token)
                    # If there is a $ after the word, add it back
                    # print('orig_idx: ', orig_idx)
                    punct_idx = orig_idx + len(token)
                    if punct_idx >= len(original_text): 
                        continue
                    if original_text[punct_idx] not in space_chars:
                        # result += '$'
                        changes.append(punct_idx)
        # print('Punctuated text')
        # print(result)
        offset = 0
        original_text2 = copy.deepcopy(original_text)
        for pos in changes:
            pos += offset
            original_text2 = original_text2[:pos] + "$$" + original_text2[pos:]
            offset += 2
        
        output_expected.append(original_text2)

        # with open(args.out_file, 'w', encoding='utf-8') as f:
        #     f.write(result + "\n")
        #     f.write('Changes: ' + str(changes) + "\n")
        #     f.write('original_text: ' + original_text + "\n")
        #     f.write('original_text2: ' + original_text2 + "\n")
    
    # Save for debugging
    p_file.close()
    preddf["punct"] = output_expected
    preddf["diff"] = preddf[incol] != preddf["punct"]
    import Levenshtein
    edit_distance = Levenshtein.distance
    df = preddf
    df['Levenstein'] = df.apply(lambda row: edit_distance(row[incol], row['punct']), axis=1)
    # Print sum and mean of Levenstein distance
    print("Sum of Levenstein distance: ", df['Levenstein'].sum())
    print("Mean of Levenstein distance: ", df['Levenstein'].mean())
    print("Sum of differences: ", df['diff'].sum())
    # preddf.to_csv(f"preds/data_debug{suffix}.csv", index=False)
    preddf.to_csv(debug_file, index=False)
    # Prepare the submission file
    preddf[incol] = preddf["punct"]
    preddf = preddf.drop(columns=["punct", "diff", "Levenstein"])
    preddf.to_csv(submit_file, index=False)


if __name__ == '__main__':
    inference()
