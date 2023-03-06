import pandas as pd
from normalizer import normalize
from tqdm import tqdm
from pathlib import Path


# Make ready for model input
if 0:
  file = './Aug2/data_v2_processed_20000_with_error.csv'
  outfile_text = Path(file).parent/(Path(file).stem + '_text.csv')
  outfile_norm = Path(file).parent/(Path(file).stem + '_text_norm.csv')

  # print(outfile_text, outfile_norm); exit()

  df = pd.read_csv(file)
  df.rename(columns={'correct_sentence': 'text'}, inplace=True)

  df = df[['text']]
  df.to_csv(outfile_text, index=False)

  df['text'] = df['text'].apply(normalize)
  df.to_csv(outfile_norm, index=False)



if 0:
  # Clean up
  file = './Aug2/data_v2_processed_20000_with_error.csv'
  pred_file = './Aug2/pred/spans-pred_test_checkpoint-12000_spell.csv'
  outfile_pred = Path(file).parent/(Path(file).stem + '_clean.csv')


  df_folds = pd.read_csv(r"DataSetFold1_u.csv/DataSetFold1_u.csv")
  df_fold2 = pd.read_csv(r"DataSetFold2.csv/DataSetFold2.csv")
  df_folds = df_folds.append(df_fold2)

  # Find the m length of the sentence column
  min_len = df_folds['sentence'].str.len().min()
  max_len = df_folds['sentence'].str.len().max()
  print(min_len, max_len)

  # Drop row of file if the nth row of pred_file is 0
  df = pd.read_csv(file)
  pred = pd.read_csv(pred_file)

  error_mask = ~pred['Expected'].str.contains('\$')
  english_mask = ~df['correct_sentence'].str.contains('[a-zA-Z0-9]')
  length_mask = (df['correct_sentence'].str.len() >= min_len) 
  length_mask &= (df['correct_sentence'].str.len() <= max_len)

  mask = error_mask & english_mask & length_mask
  print(sum(mask))
  df = df[mask]
  df = df[['sentence', 'gt']]
  df.to_csv(outfile_pred, index=False)











