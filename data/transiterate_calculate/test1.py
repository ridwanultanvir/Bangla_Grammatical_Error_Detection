import pandas as pd
import nltk

# Load the processed data csv file into a pandas dataframe
df_processed = pd.read_csv('data_v2_processed_500.csv')

# Load the en_bn.csv file into another dataframe
df_en_bn = pd.read_csv('en_bn.csv')

# Create a set of all words in the 'bangla' column
bangla_words = set(df_en_bn['bangla'].str.lower())

# Tokenize each sentence in the 'correct_sentence' column using NLTK
df_processed['tokens'] = df_processed['correct_sentence'].apply(nltk.word_tokenize)

# Find the sentences that match any words in the 'bangla' column
df_matched = df_processed[df_processed['tokens'].apply(lambda x: any(word.lower() in bangla_words for word in x))]

# Save only the 'correct_sentence' column in the 'test1.csv' file
df_matched['correct_sentence'].to_csv('test1.csv', index=False)
