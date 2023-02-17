if __name__ == "__main__":
  import pandas as pd
  # Read data\DataSetFold1_u.csv\DataSetFold1_u.csv and data\DataSetFold1.csv\DataSetFold1.csv
  df = pd.read_csv(r"DataSetFold1_u.csv/DataSetFold1_u.csv")
  # Find the rows where the gt and sentence columns are not equal
  df = df[df["gt"] == df["sentence"]]
  print(df)
  # important_punctuations = ['O', '।', '?', '!', ","]
  important_punctuations = ['O', ",", "?", "।"]
  # important_punctuations = ['O', ","]
  punctuation_dict = {',': 'COMMA', '।': 'PERIOD', '?': 'QUESTION'}
  punctuation_dict = {',': 'COMMA', '।': 'PERIOD', '?': 'QUESTION'}
  # important_punctuations = ['O', ",", "-"]
  # Create index dict for important_punctuations
  index_dict = {x: i for i, x in enumerate(important_punctuations)}
  # Create reverse index dict for important_punctuations
  reverse_index_dict = {i: x for i, x in enumerate(important_punctuations)}
  # Save sentences in a text file
  with open("DataSetFold1_u_clean_punc.txt", "w", encoding="utf-8") as f:
    for index, row in df.iterrows():
      # If last character is not in important_punctuations
      # if row["sentence"][-1] not in important_punctuations:
      f.write(row["sentence"] + "\n")

  # Iterate over the rows of the dataframe
  file = open("punc_class_dataset_4class.txt", "w", encoding="utf-8")
  for index, row in df.iterrows():
    # Iterate over the characters of the sentence column
    for token in row["sentence"].split():
      # print(token)
      # If the character is not in important_punctuations
      if token[-1] not in important_punctuations:
        output = token + " " + "O"
      else:
        if len(token) == 1:
          continue
        # output = token[:-1] + " " + token[-1]
        output = token[:-1] + " " + punctuation_dict[token[-1]]
      # file.write(output + "\n")
      file.write(output + "\t")
    
    # file.write("\n")
    # break
  
  file.close()
