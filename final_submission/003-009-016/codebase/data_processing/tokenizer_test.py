import pandas as pd
from tqdm import tqdm
from normalizer import normalize

df = pd.read_csv(r"DataSetFold1_u.csv/DataSetFold1_u.csv")
df2 = pd.read_csv(r"DataSetFold2.csv/DataSetFold2.csv")
df = df.append(df2)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('csebuetnlp/banglabert')

# import pdb; pdb.set_trace()
wordlist = '../dcspell/data/two_dictwords.csv'
df_words = pd.read_csv(wordlist)
words = df_words['word'].tolist()

from tqdm import tqdm

with open('processed_data/missing_words.txt', 'w', encoding='utf-8') as f:
  for word in tqdm(words):
    tokens = tokenizer.tokenize(word)
    if len(tokens) == 1 and tokens[0] == '[UNK]' and word != '[UNK]':
      tokenizer.add_tokens(word)
      f.write(word + '\t')


# manuals = ['’', '‘', '”', '“', '…', '–']
# for manual in manuals:
#   tokens = tokenizer.tokenize(word)
#   if len(tokens) == 1 and tokens[0] == '[UNK]' and word != '[UNK]':
#     tokenizer.add_tokens(word)


# Normalize
# df["sentence"] = df["sentence"].apply(lambda x: normalize(x))

# Tokenize df and save to a csv file
# df["tokens"] = df["sentence"].apply(lambda x: ' '.join(tokenizer.tokenize(x)))

# df.to_csv("processed_data/DataSetFold12_tokens_dict2_man.csv", index=False)



