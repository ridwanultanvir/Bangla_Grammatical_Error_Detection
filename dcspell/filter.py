import pandas as pd
from tqdm import tqdm
import re

corpus = pd.read_csv('data/corpus.csv', encoding='utf-8')
dictionary = pd.read_csv('data/df_all_words.csv', encoding='utf-8')
dictionary2 = pd.read_csv('data/BengaliWordList_439.txt', encoding='utf-8', header=None, names=["word"])
print("Corpus shape: ", corpus.shape)
# out_file = "data/bad_words.csv"
out_file = "data/bad_words_notd.csv"

error_types_bad = [
  "Typo Deletion", # This is creating a lot of problems
  # "Split-word Error (Random)",
  # "Split-word Error (Left)",
  # "Split-word Error (Right)",
  # "Split-word Error (both)", 
]
# Find good error types from ErrorType column
error_types_good = set(corpus["ErrorType"].tolist())
error_types_good = error_types_good - set(error_types_bad)
error_types_good = list(error_types_good)
print("Good error types: ", error_types_good)

# Filter rows where ErrorType is not error_types_bad with pandas 
corpus_df = corpus[corpus["ErrorType"].isin(error_types_good)]
# Reset index
corpus_df.reset_index(drop=True, inplace=True)
error_types_good = set(corpus["ErrorType"].tolist())
print("Good error types: ", error_types_good)

# Filter rows where Errortype is not error_types_bad 
# for i, errortype in tqdm(enumerate(corpus["ErrorType"].tolist()), total=len(corpus)):
#   if errortype in error_types_bad:
#     corpus.drop(i, inplace=True)

corpus = corpus_df[["Error"]]

# Extract and concat two dictionaries
words = dictionary[["word"]]
words2 = dictionary2[["word"]]
# Pandas concat as a union
words = pd.concat([words, words2], ignore_index=True)
words = words.drop_duplicates()

# Save the dictionary in a csv file
words.to_csv("data/two_dictwords.csv", index=False)
exit()

word_regex = "|".join(words.values.flatten().tolist())
print("Word regex: ", word_regex[:100])

# exit()


# Check if all the words in corpus are in the dictionary
bad_words = []
bar = tqdm(enumerate(corpus["Error"].tolist()), total=len(corpus))
for i, word in bar:
  # if not re.search(word_regex, word): # If not found in dictionary
  if not re.fullmatch(word_regex, word): # If not found in dictionary
    # print(word)
    bad_words.append(word)

  if i % 1000 == 0:
    # Save the bad words in a csv file with header Error
    df = pd.DataFrame(bad_words, columns = ['Error'])
    df.to_csv(out_file, index=False)
    bar.set_postfix_str(corpus_df["ErrorType"][i])


# Save the bad words in a csv file with header Error
df = pd.DataFrame(bad_words, columns = ['Error'])
df.to_csv(out_file, index=False)

