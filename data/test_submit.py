if __name__ == "__main__":
  import pandas as pd
  # Read test.csv into a pandas dataframe
  df = pd.read_csv("test.csv")
  df.columns = ["Expected"]
  df.index += 1
  print(df.head())
  # Save the dataframe to a csv file start index is 1, index=True
  df.to_csv("test_results.csv", index=True, index_label="Id")