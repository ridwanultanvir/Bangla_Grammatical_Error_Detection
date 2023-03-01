import pandas as pd
space_chars = ['।', '?', '!', ","]
def replace_fixed(x):
  for char in space_chars:
    x = x.replace(" "+char, "$ $"+char)
  return x

end_chars = ['।', '?', '!', 
          # '$'
    ]

def replace_end(x):
  if len(x) == 0:
    return x
  # Find the last character position from the end of the string which is not a space
  # last_char_pos = len(x) - 1
  # while x[last_char_pos] == " ":
  #   last_char_pos -= 1
  # Find the last character position from the end of the string which is not a space using regex
  import re
  last_char_pos = re.search(r"\s+$", x)
  if last_char_pos is None:
    return x if x[-1] in end_chars else x + "$$"
  last_char_pos = last_char_pos.start()

  # If the last character is not in end_chars then add a  $$
  return x if x[last_char_pos] in end_chars else x[:last_char_pos+1] + "$$" + x[last_char_pos+1:]



  # return x if x[-1] in end_chars else x + "$$"

if __name__ == "__main__":
  # df = pd.read_csv(r"DataSetFold1.csv/DataSetFold1.csv")
  # df = pd.read_csv(r"test_results_tsd_bert_modi.csv")
  # df = pd.read_csv(r"test_results_tsd_bert.csv")

  # df["Expected"] = df["Expected"].apply(replace_fixed)

  # # Find the last character of gt column whose frequency is greater than 20
  # # df["last_char"] = df["Expected"].apply(lambda x: x[-1])
  # # stat = df["gt"].apply(lambda x: x[-1]).value_counts()
  # stat = df["Expected"].apply(lambda x: x[-1]).value_counts()
  # # From stat filter out the characters whose frequency is greater than 20
  # stat = stat[stat >20]
  # print(stat)
  # # Create a list of characters whose frequency is greater than 20
  # chars = list(stat.index)
  # print(chars)
  # end_chars = ['।', '?', '!', 
  #         # '$'
  # ]

  # # If the last character of the text column is not in end_chars then add a  $$
  # df["Expected"] = df["Expected"].apply(replace_end)
  # # Save the dataframe to a csv file
  # df.to_csv("test_results_tsd_bert_modi_end.csv", index=False)
  space_end = "তিনি প্রাকৃতিক বিজ্ঞান, আলোকচিত্রবিদ্যা, ধর্ম বিষয়ে আগ্রহী ছিলেন।"
  fix = replace_end(space_end)
  print(fix, len(fix), len(space_end))

  space_end = "ওর ফাঁসি চাই,না হলে ওদের বিয়ে দিয়ে দিতে হবে "
  fix = replace_end(space_end)
  print(fix, len(fix), len(space_end))
  space_end = "ওর ফাঁসি চাই,না হলে ওদের বিয়ে দিয়ে দিতে হবে"
  fix = replace_end(space_end)
  print(fix, len(fix), len(space_end))

  

  

  
