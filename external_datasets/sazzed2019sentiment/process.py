'''
cd external_datasets/sazzed2019sentiment
python process.py
'''
import pandas as pd

positive = "gt/all_positive_8500.txt"
negative = "gt/all_negative_3307.txt"
pos_out = "processed/all_p8500_n3307.csv"

df_pos = pd.read_csv(positive, header=None, encoding="utf-8", sep="\n")
df_pos.columns = ["text"]
df_pos["label"] = "p"
df_neg = pd.read_csv(negative, header=None, encoding="utf-8", sep="\n")
df_neg.columns = ["text"]
df_neg["label"] = "n"
# Concatenate to df_all
df_all = pd.concat([df_pos, df_neg], axis=0)


# Strip whitespace from left
df_all["text"] = df_all["text"].str.lstrip()
print(df_all.head())
df_all.to_csv(pos_out, index=False)


