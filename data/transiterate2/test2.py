import pandas as pd

# Define the phonetic dictionary
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
def pure_char(s):
        ret=''
        for c in s:
            if ord(c) == 9676:
                continue
            ret+=c
        return ret
# Load the original CSV file with the matched sentences and tokens
df_matched = pd.read_csv('test1.csv')

# Replace the characters in the matched token based on the phonetic_dict
df_matched['gt'] = df_matched.apply(lambda row: row['correct_sentence'].replace(row['matched_token'], ''.join([pure_char(phonetic_dict.get(c, c)) for c in row['matched_token']])), axis=1)
# df_matched['gt'] = df_matched.apply(lambda row: row['correct_sentence'].replace(
#     row['matched_token'], 
#     ''.join([phonetic_dict.get(pure_char(c), c) for c in row['matched_token']])
# ), axis=1)

# Create a new CSV file with only the corrected sentences
df_matched[['correct_sentence', 'gt','matched_token']].to_csv('test2.csv', index=False)
