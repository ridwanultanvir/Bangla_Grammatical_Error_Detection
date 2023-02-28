import pandas as pd

def transiterate_cnt():
    # import pandas as pd
    from nltk.tokenize import word_tokenize

    # Read the CSV files into dataframes
    # df1 = pd.read_csv('data_v2_processed_500.csv')
    
    csv_file = '../../../archive/data_v2/data_v2_processed_20000.csv'
    df1 = pd.read_csv(csv_file)
    df2 = pd.read_csv('en_bn.csv')

    # Tokenize the words in the 'correct_sentence' column of df1
    words1 = set()
    for sentence in df1['correct_sentence']:
        # words1.update(word_tokenize(sentence.lower()))
        words1.update(word_tokenize(sentence))

    # # Tokenize the words in the 'bangla' column of df2
    words2 = set()
    for sentence in df2['bangla']:
        # words2.update(word_tokenize(sentence.lower()))
        words2.update(word_tokenize(sentence))

    # Find the intersection of the two sets
    common_words = words1.intersection(words2)

    # Count the number of words in the intersection set
    num_common_words = len(common_words)
    
    
    print(list(words1)[:5])
    print(list(words2)[:5])
    print("Number of common words: ", num_common_words)

    
    return common_words
    # print(f'The number of words in data_v2_processed_500.csv that match with the bangla column in en_bn.csv is {num_common_words}.')



common_words = transiterate_cnt()

import csv 

# Dump the common_words set into a CSV file
with open('common_words_full.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['word'])
    for word in common_words:
        writer.writerow([word])