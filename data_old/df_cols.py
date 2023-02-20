if __name__ == "__main__":
  import pandas as pd
  df = pd.read_csv(r"DataSetFold1.csv/DataSetFold1.csv")
  # only keep the text
  df = df[["gt"]]
  # Save the dataframe to a csv file
  df.to_csv("DataSetFold1_gt.csv", index=False)