# Find the first occurence of a token in a sentence, and replace token with a new token
def find_nth_occurrence(text, word, n):
  start = -1
  for i in range(n):
      start = text.find(word, start + 1)
      if start == -1:
          return -1
  return start

from collections import defaultdict

# Read bad words csv pandas
import pandas as pd
# bad_words = pd.read_csv("data/bad_words.csv")

bad_words = pd.read_csv("data/bad_words_notd_wiki_man.csv")

# trainfile = r"../data/submissions/test_results_bn3500_bnlargefull1_int.csv"
# trainfile = r"../data/three_class/bert_token_3cls_banglabert/spans-pred_test_checkpoint-3000.csv"
# trainfile = r"../data/three_class/bert_token_3cls_bertlarge7000_banglabert3000_int.csv"
# trainfile = r"../data/three_class/bert_token_3cls_bertlarge_fullfold1/spans-pred_test_checkpoint-9000.csv"
# trainfile = r"../data/three_class/bert_token_3cls_bertlarge/spans-pred_test_checkpoint-7000.csv"
# trainfile = r"../data/three_class/bert_token_3cls_bertlarge/spans-pred_test_checkpoint-3000.csv"
trainfile = r"../data/pred/3_cls/bert_large/spans-pred_test_checkpoint-6000.csv"

from pathlib import Path
pred_file_p = Path(trainfile)
out_file = pred_file_p.parent/(pred_file_p.stem + '_spell.csv')

col = "Expected"
train = pd.read_csv(trainfile, encoding='utf-8')
bad_words = bad_words.values.flatten().tolist()
# Create regex pattern by joining all bad words
import re
from tqdm import tqdm
pattern = re.compile("|".join(bad_words))
from ner import NER
model_path = "model/bn_ner.pkl"
bn_ner = NER(model_path=model_path)

# Iterate over all sentences
outputs = []
for idx, sentence in tqdm(enumerate(train[col].tolist()), total=len(train)):
  # Find all the bad words in the sentence
  # bad_words_in_sentence = pattern.fullmatch(sentence)
  # bad_words_in_sentence = pattern.findall(sentence)
  bad_words_in_sentence = []
  # sentence = "মতিঝিলে সরিষে ইলিশ রেস্তোরাঁয় রিভিউ করতে পারেন$ $, খুবই ভালো পরিবেশ$ $, আর সরিষে মতিঝিলে"
  # Find the ner tags for the sentence
  res = bn_ner.tag(sentence)
  ner_tags = defaultdict(lambda: "O")
  for r in res:
    ner_tags[r[0]] = r[1]
  for word in sentence.split():
    if ner_tags[word] != "O": continue
    if pattern.fullmatch(word):
      bad_words_in_sentence.append(word)

  # Replace bad words with $badword$
  output = sentence
  bad_words_in_sentence = list(set(bad_words_in_sentence))
  for bad_word in bad_words_in_sentence:
  # if 1:
    # sentence = sentence.replace(bad_word, "$" + bad_word + "$")
  # Print the sentence
  # print(sentence)

    frequency = defaultdict(int)

    # token = "ঘৃনা" 
    token = bad_word
    print(f"Token: {token}")
    replace_token = "$" + token + "$"
    # sentence = "কিন্তু এখন আপনার জন্য $ঘৃনা$ আমার মনে প্রচন্ড ঘৃনা কাজ জানেন।"
    # sentence = "কিন্তু এখন আপনার জন্য ঘৃনা আমার মনে প্রচন্ড ঘৃনা কাজ জানেন।"

    
    for word in sentence.split():
      frequency[word] += 1
      if word == replace_token or word == token + "$" or word == "$" + token: 
        # Full or partial match
        frequency[token] += 1
      if word == token:
        index_out = find_nth_occurrence(output, word, frequency[word])
        print(f"Found {word} at index {index_out}")
        output = output[:index_out] + replace_token + output[index_out+len(word):]

    print("Output: ", output)
    # break
  # exit()


  outputs.append(output)


# Save output to Expected column
train["Expected"] = outputs
print("Saving to ", out_file)
# train.to_csv("submissions/test_results_bn3500_bnlargefull1_int_spell.csv", index=False)
train.to_csv(out_file, index=False)



