import pandas as pd

phonetic_dict = {
            'ঋ': 'রি',
            'ঐ': 'অই',
            'ঔ': 'অউ',
            'খ': 'ক',
            'ঙ': '◌ং',
            'ঝ': 'জ',
            'ঠ': 'ট',
            'ড': 'দ',
            'ঢ': 'ড',
            'থ': 'ত',
            'দ': 'ড',
            'ধ': 'দ',
            'ভ': 'ব',
            'য': 'জ',
            'ৎ': 'ত',
            'ং': 'ঙ',
            'ঃ': 'হ',
            '‍ঁ': '',
            'ৈ': 'ই',
            'ৌ': 'উ',
            '‍ঢ়': 'র',
            'ড়': 'র',
            'ৃ': 'রি',
            'অ': 'ও',
            'ই': 'ঈ',
            'উ': 'ঊ',
            'চ': 'ছ',
            'ট': 'ত',
            'ড': 'দ',
            'ন': 'ণ',
            'য': 'জ',
            'শ': 'স',
            'ি': '◌ী',
            'ু': '◌ূ'
        }


common_words = pd.read_csv('common_words.csv')['word'].tolist()
def apply_phonetic_dict(sentence):
    for word in common_words:
        if word in sentence:
            # Add $ before and after the word
            sentence = sentence.replace(word, f'${word}$')
            # Apply the phonetic dictionary
            for char, phonetics in phonetic_dict.items():
                for phonetic in phonetics.split(','):
                    sentence = sentence.replace(char, phonetic)
    return sentence

# Apply the function to the data




def replace_with_phonetic_equivalent(filename, words_filename):
    # Load the words file into a list
    with open(words_filename, 'r', encoding='utf-8') as f:
        words = f.read().splitlines()

    # Load the data file into a DataFrame
    data = pd.read_csv(filename)

    # Loop through each word in the list and replace any matching words in the correct_sentence column
    for word in words:
        phonetic_equivalent = ''.join(phonetic_dict.get(c, c) for c in word) # convert word to phonetic equivalent
        
        # data['gt'] = data['correct_sentence'].str.replace(word, phonetic_equivalent)
        data['gt'] = data['correct_sentence'].apply(apply_phonetic_dict)

    # Write the updated DataFrame back to the file
    data.to_csv('trans1.csv', index=False)

csv_file = '../../../archive/data_v2/data_v2_processed_500.csv'
replace_with_phonetic_equivalent(csv_file, 'common_words.csv')