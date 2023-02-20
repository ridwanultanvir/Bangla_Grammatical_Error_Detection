
'''
cd BanglaNLG/seq2seq
python create_dataset.py
'''
import pandas as pd

'''
train_df = pd.read_csv("../../data/processed_data/train.csv")
valid_df = pd.read_csv("../../data/processed_data/valid.csv")

train_df = train_df[["sentence", "gt"]]
valid_df = valid_df[["sentence", "gt"]]

# train_df[:2].to_json(orient='records', lines=True)

with open('banglaged/train.jsonl', 'w', encoding='utf-8') as f:
    f.write(train_df.to_json(orient='records', lines=True, force_ascii=False))

with open('banglaged/validation.jsonl', 'w', encoding='utf-8') as f:
    f.write(valid_df.to_json(orient='records', lines=True, force_ascii=False))
'''

test_df = pd.read_csv("../../data/processed_data/test_tsd.csv")
test_df.columns = ["sentence"]
test_df["gt"] = [""] * len(test_df)
# SAve as jsonl
with open('banglaged/test.jsonl', 'w', encoding='utf-8') as f:
    f.write(test_df.to_json(orient='records', lines=True, force_ascii=False))

  

