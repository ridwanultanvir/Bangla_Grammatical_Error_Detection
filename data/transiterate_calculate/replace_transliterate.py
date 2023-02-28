import pandas as pd


phonetic_dict = {
        'ঋ' : ['রি'],
        'ঐ': ['অই'],
        'ঔ': ['অউ'],
        'খ': ['ক'],
        'ঙ': ['◌ং'],
        'ঝ': ['জ'],
        'ঠ' : ['ট','ত'],
        'ড' : ['দ'],
        'ঢ': ['ড'],
        'থ': ['ত','ট'],
        'দ' : ['ড'],
        'ধ' : ['দ','ড'],
        'ভ' : ['ব'],
        'য' : ['জ'],
        'ৎ' : ['ত'],
        'ং' : ['ঙ'],
        'ঃ': ['হ'],
        '‍ঁ' : [''],
        'ৈ'  :['ই'],
        'ৌ':['উ'],
        '‍ঢ়': ['র'],
        'ড়' : ['র'],
        'ৃ' : ['রি'],
        'অ' : ['ও'],
        'ই':['ঈ'],
        'উ' : ['ঊ'],
        'চ': ['ছ'],
        'ট' :['ত'],
        'ড': ['দ'],
        'ন': ['ণ'],
        'য' : ['জ'],
        'শ' : ['স','ষ'],
        'ি' : ['◌ী'],
        'ু' : ['◌ূ']
    }


def replace_with_phonetic_equivalent(filename, words_filename):
    # Load the words file into a list
    with open(words_filename, 'r', encoding='utf-8') as f:
        words = f.read().splitlines()

    # Load the data file into a DataFrame
    data = pd.read_csv(filename)

    # Loop through each word in the list and replace any matching words in the correct_sentence column
    for word in words:
        phonetic_equivalent = ''.join(phonetic_dict.get(c, c) for c in word) # convert word to phonetic equivalent
        data['gt'] = data['correct_sentence'].str.replace(word, phonetic_equivalent)

    # Write the updated DataFrame back to the file
    data.to_csv('trans1.csv', index=False)

csv_file = '../../../archive/data_v2/data_v2_processed_20000.csv'
replace_with_phonetic_equivalent(csv_file, 'common_words.csv')