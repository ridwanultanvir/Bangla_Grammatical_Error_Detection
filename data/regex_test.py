import pandas as pd 

csv = 'DataSetFold1_u.csv/DataSetFold1_u.csv'

df = pd.read_csv(csv, encoding='utf-8')

# Write gt column to file
with open('gt.txt', 'w', encoding='utf-8') as f:
  for row in df['gt']:
    f.write(row + '\n')

