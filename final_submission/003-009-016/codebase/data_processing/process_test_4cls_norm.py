# test_file = "../ninth_place_tsd/data/test_tsd.csv"
# out_file = "../ninth_place_tsd/data/test_tsd_norm.csv"
from normalizer import normalize
from tqdm import tqdm
import pandas as pd
import re

# dataset = pd.read_csv(test_file)
# apply normalization to sentences
# dataset['text'] = dataset['text'].apply(normalize)

# save the normalized sentences to a file
# dataset.to_csv(out_file, index=False)

# subfile = './DataSetFold2.csv/SampleSubmission.csv'
# sub = pd.read_csv(subfile)
# sub['Expected_norm'] = sub['Expected'].apply(normalize)
# sub['norm_diff'] = sub['Expected'] != sub['Expected_norm']

col_gen = 'text'

subfile = './DataSetFold2.csv/test.csv'
col = 'text'
out_file = './DataSetFold2.csv/test_norm.csv'
subfile = './data_generator/data_v2_processed_20000.csv'
col = 'correct_sentence'
out_file = './Aug1/data_v2_processed_20000.csv'


sub = pd.read_csv(subfile)
sub[col_gen] = sub[col]
# sub['text_norm'] = sub[col].apply(normalize)
# sub['norm_diff'] = sub[col] != sub['text_norm']

# sub[col_gen] = sub[col].apply(normalize)
sub = sub[[col_gen]]

sub.to_csv(out_file, index=False)


