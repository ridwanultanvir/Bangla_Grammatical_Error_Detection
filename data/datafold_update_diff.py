if __name__ == "__main__":
  import pandas as pd
  # Read data\DataSetFold1_u.csv\DataSetFold1_u.csv and data\DataSetFold1.csv\DataSetFold1.csv
  df = pd.read_csv(r"DataSetFold1_u.csv/DataSetFold1_u.csv")
  df1 = pd.read_csv(r"DataSetFold1.csv/DataSetFold1.csv")
  # Find the rows where the gt columns are not equal
  df = df[df["gt"] != df1["gt"]]
  # Save the dataframe to a csv file
  df.to_csv("DataSetFold1_u_diff.csv", index=False)