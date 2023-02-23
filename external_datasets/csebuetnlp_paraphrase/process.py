import pandas as pd

train_json = "gt/train.jsonl"
test_json = "gt/test.jsonl"
valid_json = "gt/validation.jsonl"
outfile = "processed/csebuetnlp_paraphrase.csv"

# Read all json files
df_train = pd.read_json(train_json, lines=True)
df_test = pd.read_json(test_json, lines=True)
df_valid = pd.read_json(valid_json, lines=True)
print(df_train.head())

# Concatenate to df_all
df_all = pd.concat([df_train, df_test, df_valid], axis=0)
print(df_all.head())

# Rename text to text
df_all = df_all.rename(columns={"source": "text"})
print(df_all.head())

# Save to csv
df_all.to_csv(outfile, index=False)

# Small version
df_all = df_all.sample(n=5000, random_state=42)
df_all.to_csv("processed/csebuetnlp_paraphrase_5k.csv", index=False)

