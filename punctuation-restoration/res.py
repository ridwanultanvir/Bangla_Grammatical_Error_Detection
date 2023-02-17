original_str = "The quick brown fox jumps quick over the lazy dog."
insert_positions = [4, 10, 16] # positions to insert substrings
substrings = ['red', 'blue', 'green'] # substrings to insert
new_str = original_str

# iterate over the positions and substrings and insert them into the new string
# offset = 0
# for pos in insert_positions:
#     pos += offset
#     new_str = new_str[:pos] + "$$" + new_str[pos:]
#     offset += 2

# print(new_str)

def find_nth_occurrence(text, word, n):
  start = -1
  for i in range(n):
      start = text.find(word, start + 1)
      if start == -1:
          return -1
  return start

print(find_nth_occurrence(original_str, 'quick', 3))

# read punctuation-restoration/preds/test_results_bn3500_bnlargefull1_int_punct_debug.csv
import pandas as pd

# df = pd.read_csv("preds/test_results_bn3500_bnlargefull1_int_punct_debug.csv")
# # Calculate the levenstein distance between the original text and the predicted text
# import Levenshtein
# edit_distance = Levenshtein.distance
# df['Levenstein'] = df.apply(lambda row: edit_distance(row['Expected'], row['punct']), axis=1)
# # Print sum and mean of Levenstein distance
# print("Sum of Levenstein distance: ", df['Levenstein'].sum())
# print("Mean of Levenstein distance: ", df['Levenstein'].mean())
# # Save the dataframe to a csv file
# df.to_csv("preds/test_results_bn3500_bnlargefull1_int_punct_debug.csv", index=False)
import pandas as pd

preddf = pd.read_csv("preds/test_results_bn3500_bnlargefull1_int_punct_gedtrain_submit_ 0.10_ep46.csv")
preddf = preddf.drop(columns=["punct", "diff", "Levenstein"])
preddf.to_csv(f"preds/test_results_bn3500_bnlargefull1_int_punct_gedtrain_submit_ 0.10_ep46.csv", index=False)


  