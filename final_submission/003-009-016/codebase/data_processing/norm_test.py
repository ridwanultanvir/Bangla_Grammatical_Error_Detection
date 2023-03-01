# from normalizer_local import normalize
from normalizer import normalize
from tqdm import tqdm

from nltk.metrics.distance import edit_distance_align

def g():
  from transformers import AutoTokenizer
  import re
  import pandas as pd
  tokenizer = AutoTokenizer.from_pretrained('csebuetnlp/banglabert')


  dataset_path = f"DataSetFold1_u.csv/DataSetFold1_u.csv"
  dataset_path2 = f"DataSetFold2.csv/DataSetFold2.csv"

  dataset = pd.concat([pd.read_csv(dataset_path), pd.read_csv(dataset_path2)], axis=0).reset_index()

  gt = dataset[['gt']]

  unknowns = []
  file = open('unknowns.txt', 'w', encoding='utf-8')
  for sentence in tqdm(gt['gt'], total=len(gt)):
    # Reverse the sentence
    # sentence = sentence[::-1]
    sentence_n = normalize(sentence)
    x = edit_distance_align(sentence, sentence_n, substitution_cost=10)
    if sentence != sentence_n:
      for i, a in enumerate(x):
        if a[0] != 0 and a[1] != 0:
          p = sentence[a[0]-1]
          q = sentence_n[a[1]-1]
          if (p == 'য়' and q == ' ়'[-1]) or (p == 'ড়' and q == ' ়'[-1]) \
            or (p == 'ঢ়' and q == ' ়'[-1]) or (p == 'র' and q == ' ়'[-1]):
            if i - 1 >= 0:
              x[i-1] = (x[i-1][0]+1, x[i-1][1])
          # and sentence[a[0]-1] != sentence_n[a[1]-1] \          
          # print(sentence[a[0]-1], sentence_n[a[1]-1], file=file, flush=True)
          if (p == '—' and q == ' ') or (p == '…' and q == '.'):
            if i - 1 >= 0:
              x[i-1] = (x[i-1][0]+1, x[i-1][1])
            if i - 2 >= 0:
              x[i-2] = (x[i-2][0]+1, x[i-2][1])
      
      for i, a in enumerate(x):
        if a[0] != 0 and a[1] != 0:
          p = sentence[a[0]-1]
          q = sentence_n[a[1]-1]        
          print(p, q, file=file, flush=True)
    # exit()
    # print(x)
    sentence = sentence_n
    # sentence, ind = fix_text_with_indices(sentence)
    tokens = tokenizer.tokenize(sentence)
    for token in tokens:
      if token == '[UNK]':
        unknowns.append(sentence + ", " + repr(tokens))

  print(len(unknowns))
  file.close()
  # print('\n'.join(unknowns))

from ftfy import fix_text, fix_and_explain, fix_text_segment
import pdb



# text = 'সব চেয়ে'*1000
text = 'ড়'
print(text, len(text))
fix = normalize(text)
# print(n_text, len(n_text))
# fix = fix_text_segment(
#         text,
#     )
# Print the indices of the characters that were changed by ftfy

# fix, ind = fix_text_with_indices(text)
print(fix, len(fix))
# print(ind)
# print([ord(c) for c in text], [ord(c) for c in fix])

# x = edit_distance_align(text, fix)
# print(x)
g()
# pdb.set_trace()



