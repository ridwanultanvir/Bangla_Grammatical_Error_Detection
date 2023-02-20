import pandas as pd

old_df = "DataSetFold1.csv/DataSetFold1.csv"
old_df = pd.read_csv(old_df)
new_df = "DataSetFold1_u.csv/DataSetFold1_u.csv"
new_df = pd.read_csv(new_df)

# Join the two dataframes by the column "sentence"
df = pd.merge(old_df, new_df, on="sentence")
print(df.columns)
print(df.head())

df["diff"] = df["gt_x"] != df["gt_y"]

# drop sentence column
df = df.drop(columns=["sentence"])
print(df["diff"].value_counts())
# Keep columns where diff is True
df = df[df["diff"] == True]


# save the dataframe to a csv file
df.to_csv("Datafold1_diff.csv")


