from ner import NER
from tqdm import tqdm

model_path = "model/bn_ner.pkl"
bn_ner = NER(model_path=model_path)
# text = "সে ঢাকায় থাকে।" # or you can pass token list
# res = bn_ner.tag(model_path, text)

import pandas as pd
bad_words = pd.read_csv("data/bad_words.csv", encoding='utf-8')

sent = "জেনিয়া জাহেদের কথা শুনে হাসতে থাকে"
res = bn_ner.tag(sent)
print(res)
sent = "জেনিয়া জাহেদের "
res = bn_ner.tag(sent)
print(res)
exit()


wordlist = bad_words.values.flatten().tolist()
wordlist = " ".join(wordlist)
res = bn_ner.tag(model_path, wordlist)
file = open("data/bad_words_ner.txt", "w", encoding="utf-8")
for i, r in tqdm(enumerate(res), total=len(res)):
  if r[1] != "O":
    # file.write(r[0] + ", " + r[1] + "\n")
    # file.flush()
    # Drop ith word from bad_words
    bad_words.drop(bad_words.index[i], inplace=True)

file.close()

bad_words = pd.read_csv("data/bad_words_ner.csv", encoding='utf-8')

# for word in tqdm(wordlist):
#   print(word)
#   res = bn_ner.tag(model_path, word)
#   print(res)

