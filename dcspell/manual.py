words = [
  # '',
  # Named entities
  # 'বুশারি', 'ঢুডু', 'জেনিয়া', 'আঈনি', '',

  'ভাড়া', 'এনাল', 'ফুরি', 'যেতা', 'বাড়ি', 'চলো', 'দ্য', 'কামড়াবো',
  'যেতা', 'অনে', 'মটে', 
  # 'বালের', 
  'ব্রি'
]


import pandas as pd
bad_words = pd.read_csv("data/bad_words_ner.csv", encoding='utf-8')

# Filter rows that does **not** contain any of the words
bad_words = bad_words[~bad_words.isin(words).any(axis=1)]
# Save the dataframe 
bad_words.to_csv('data/bad_words_ner_man.csv', index=False)
