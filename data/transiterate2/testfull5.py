import pandas as pd
import nltk

# Load the processed data csv file into a pandas dataframe
csv_file = '../../../archive/data_v2/data_v2_processed.csv'
df_processed = pd.read_csv(csv_file)
print(df_processed.shape)

# Load the en_bn.csv file into another dataframe
df_en_bn = pd.read_csv('en_bn.csv')

# Create a set of all words in the 'bangla' column
bangla_words = set(df_en_bn['bangla'].str.lower())

# Tokenize each sentence in the 'correct_sentence' column using NLTK
df_processed['tokens'] = df_processed['correct_sentence'].apply(nltk.word_tokenize)

# Find the sentences that match any words in the 'bangla' column
df_matched = df_processed[df_processed['tokens'].apply(lambda x: any(word.lower() in bangla_words for word in x))]

# Find the specific 'bangla' token that matches each sentence
df_matched = df_matched.loc[:, ['correct_sentence', 'tokens']].copy()
df_matched['matched_token'] = df_matched['tokens'].apply(lambda x: next((word for word in x if word.lower() in bangla_words), None))

# Save the 'correct_sentence' and 'matched_token' columns to the 'test1.csv' file
df_matched[['correct_sentence', 'matched_token']].to_csv('testfull5.csv', index=False)

df_matched[['correct_sentence']].to_csv('testfull6.csv', index=False)
