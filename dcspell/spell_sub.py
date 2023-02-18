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
bad_words = pd.read_csv("data/bad_words.csv")
trainfile = r"../data/submissions/test_results_bn3500_bnlargefull1_int.csv"
col = "Expected"
train = pd.read_csv(trainfile, encoding='utf-8')
bad_words = bad_words.values.flatten().tolist()
# Create regex pattern by joining all bad words
import re
from tqdm import tqdm
pattern = re.compile("|".join(bad_words))

# Iterate over all sentences
outputs = []
for idx, sentence in tqdm(enumerate(train[col].tolist()), total=len(train)):
  # Find all the bad words in the sentence
  # bad_words_in_sentence = pattern.fullmatch(sentence)
  # bad_words_in_sentence = pattern.findall(sentence)
  bad_words_in_sentence = []
  for word in sentence.split():
    if pattern.fullmatch(word):
      bad_words_in_sentence.append(word)

  # Replace bad words with $badword$
  output = sentence
  for bad_word in bad_words_in_sentence:
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


  outputs.append(output)


# Save output to Expected column
train["Expected"] = outputs
train.to_csv("submissions/test_results_bn3500_bnlargefull1_int_spell.csv", index=False)



