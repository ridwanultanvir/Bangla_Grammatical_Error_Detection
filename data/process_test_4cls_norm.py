test_file = "../ninth_place_tsd/data/test_tsd.csv"
out_file = "../ninth_place_tsd/data/test_tsd_norm.csv"
from normalizer import normalize
from tqdm import tqdm
import pandas as pd
import re

dataset = pd.read_csv(test_file)
# apply normalization to sentences
# dataset['text'] = dataset['text'].apply(normalize)

# save the normalized sentences to a file
# dataset.to_csv(out_file, index=False)

# subfile = './DataSetFold2.csv/SampleSubmission.csv'
# sub = pd.read_csv(subfile)
# sub['Expected_norm'] = sub['Expected'].apply(normalize)
# sub['norm_diff'] = sub['Expected'] != sub['Expected_norm']

subfile = './DataSetFold2.csv/test.csv'
sub = pd.read_csv(subfile)
sub['text_norm'] = sub['text'].apply(normalize)
sub['norm_diff'] = sub['text'] != sub['text_norm']

sub.to_csv('./DataSetFold2.csv/test_norm.csv', index=False)


