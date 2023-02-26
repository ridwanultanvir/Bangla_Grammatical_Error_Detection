import pandas as pd

dataset = "gt/Dataset.xlsx"

out_file = "processed/maheen2022alternative.csv"

df = pd.read_excel(dataset)
# Rename Text to text
df = df.rename(columns={"Text": "text"})
print(df.head())
df.to_csv(out_file, index=False)

