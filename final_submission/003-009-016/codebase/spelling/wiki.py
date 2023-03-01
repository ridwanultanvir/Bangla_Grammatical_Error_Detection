import pandas as pd
from tqdm import tqdm


def create_wordlist():

  wikipath = "data/bnwiki-latest-all-titles"
  df = pd.read_csv(wikipath, delimiter='\t')
  print(df.head())
  # Split the title into words by punctuations
  df['words'] = df['page_title'].str.split(r'[\s\-\_\(\)\[\]\{\}\:\;\,\.\?\!/]+')
  print(df.head())
  words = []
  for i, row in tqdm(df.iterrows(), total=len(df)):
    try: words += row['words']
    except: pass

  words = set(words)
  # Remove empty strings
  words = [word for word in words if word]
  # Remove numbers
  words = [word for word in words if not word.isnumeric()]
  # Remove words with length less than 2
  words = [word for word in words if len(word) > 1]
  # Remove words with length greater than 20
  words = [word for word in words if len(word) < 20]
  # Filter word with all characters in bangla unicode range
  words = [word for word in words if all(ord(c) >= 2432 and ord(c) <= 2559 for c in word)]

  # Save words to dataframe with header title 
  df = pd.DataFrame(words, columns = ['title'])
  df.to_csv('data/bnwiki_words.csv', index=False)

# create_wordlist()

# exit()
# Read the dictionary
dictionary = pd.read_csv("data/bnwiki_words.csv")
# Convert all title to strings 
dictionary["title"] = dictionary["title"].astype(str)

regex_words = "|".join(dictionary["title"].tolist())
print("Regex words: ", regex_words[:100])

import re
# Read the bad words
bad_words = pd.read_csv("data/bad_words_notd.csv")
print("Bad words shape: ", bad_words.shape)
# Iterate over the bad words
dict_list = dictionary["title"].tolist()
for i, word in tqdm(enumerate(bad_words["Error"].tolist()), total=len(bad_words)):
  # If the word is found in the dictionary remove it from the bad words
  if re.fullmatch(regex_words, word):
    bad_words.drop(i, inplace=True)

  # # Iteartes over the dictionary
  # for title in doct_list:
  #   # If the word is found in the dictionary
  #   try:
  #     if re.fullmatch(title, word):
  #       # Remove the word from the bad words
  #       bad_words.drop(i, inplace=True)
  #       break
  #   except: 
  #     pass

# Save the bad words in a file name bad_words_wiki.csv
bad_words.to_csv("data/bad_words_notd_wiki.csv", index=False)