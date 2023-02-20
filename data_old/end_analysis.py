import pandas as pd
space_chars = ['।', '?', '!', ","]
def replace_fixed(x):
  for char in space_chars:
    x = x.replace(" "+char, "$ $"+char)
  return x

if __name__ == "__main__":
  # df = pd.read_csv(r"DataSetFold1.csv/DataSetFold1.csv")
  # df = pd.read_csv(r"test_results_tsd_bert_modi.csv")
  df = pd.read_csv(r"test_results_tsd_bert.csv")

  df["Expected"] = df["Expected"].apply(replace_fixed)

  # Find the last character of gt column whose frequency is greater than 20
  # df["last_char"] = df["Expected"].apply(lambda x: x[-1])
  # stat = df["gt"].apply(lambda x: x[-1]).value_counts()
  stat = df["Expected"].apply(lambda x: x[-1]).value_counts()
  # From stat filter out the characters whose frequency is greater than 20
  stat = stat[stat >20]
  print(stat)
  # Create a list of characters whose frequency is greater than 20
  chars = list(stat.index)
  print(chars)
  end_chars = ['।', '?', '!', 
          # '$'
    ]
  


  # If the last character of the text column is not in end_chars then add a  $$
  df["Expected"] = df["Expected"].apply(lambda x: x if x[-1] in end_chars else x + "$$")
  # Save the dataframe to a csv file
  df.to_csv("test_results_tsd_bert_modi_end.csv", index=False)

  

  

  
