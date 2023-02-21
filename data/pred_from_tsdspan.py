import pandas as pd
import os
from operator import itemgetter
from itertools import groupby
from ast import literal_eval

def get_ranges(data):
  ranges = []
  for key, group in groupby(enumerate(data), lambda x:x[0]-x[1]):
      group = list(map(itemgetter(1), group))
      if len(group) > 1:
          ranges.append(range(group[0], group[-1]))
      else:
          ranges.append(group[0])
  return ranges


space_chars = ['।', '?', '!', ","]
def replace_fixed(x):
  for char in space_chars:
    x = x.replace(" "+char, "$ $"+char)
  return x

end_chars = ['।', '?', '!', 
          # '$'
]
end_char_before = ['।', '?', '!', '.', '*']
end_before = [x+ "$" for x in end_char_before]
# temp = open("temp.txt", "w")
def replace_end(x):
  if len(x) == 0:
    return x
  # Find the last character position from the end of the string which is not a space
  # last_char_pos = len(x) - 1
  # while x[last_char_pos] == " ":
  #   last_char_pos -= 1
  # Find the last character position from the end of the string which is not a space using regex
  # Already handled
  # if x.endswith("$$$"):
  #   print(x)
  # temp.write(x + "\n")

  if x.endswith("$$$$"):
    # print("End with $$$$", x)
    return x[:-2]

  if len(x) >= 2 and x[-2:] in end_before:
    print("End with punct$", x)
    # print(x)
    return x

  if x.endswith("$$"):
    return x

  # last_char_pos = re.search(r"\s+$", x)
  # if last_char_pos is None:
  #   return x if x[-1] in end_chars else x + "$$"
  # last_char_pos = last_char_pos.start()
  # # If the last character is not in end_chars then add a  $$
  # return x if x[last_char_pos] in end_chars else x[:last_char_pos+1] + "$$" + x[last_char_pos+1:]
  return x if x[-1] in end_chars else x + "$$"

if __name__ == "__main__":
  # Read test_tsd.csv
  test_file = "test_tsd.csv"
  # pred_file = "pred/spans-pred_test.txt"
  # pred_file = "pred/spans_pred_bn_large.txt"
  # pred_file = "pred/spans_pred_xlm_base.txt"
  # pred_file = "pred/spans_pred_bn_large_745.txt"
  # pred_file = "pred/spans_pred_debertav3_large.txt"
  model = "bertlargecrf"
  suffix = "checkpoint-3000"
  # pred_file = f"pred/bertcrf/spans-pred-test.txt"
  pred_file = f"pred/{model}/spans-pred-test_{suffix}.txt"
  # pred_file = f"pred/bertcrf/spans_pred_{suffix}.txt"
  out_file = f"submissions/{model}/test_results_{model}_{suffix}.csv"

  pred_file = f"pred/spans_pred_bertcrf3000_bertlargecrf3000_union.txt"
  out_file = f"submissions/spans_pred_bertcrf3000_bertlargecrf3000_union.csv"
  
  train_check = False
  # train_check = True
  if train_check:
    test_file = "train.csv"

    # pred_file = "pred/bert_large_25ep/spans-pred_train_checkpoint-8500.txt"
    # out_file = "pred/bert_large_25ep/train.csv"
    # pred_file = "pred/bertcrf/spans-pred-train.txt"
    # out_file = "pred/bertcrf/train.csv"
    pred_file = "pred/bert/spans-pred_train_checkpoint-500.txt"
    out_file = "pred/bert/train.csv"

    # test_file = "test.csv"
    # pred_file = "pred/bert_large_25ep/spans-pred_validation_checkpoint-8500.txt"
    # out_file = "pred/bert_large_25ep/test.csv"
  

  # os.makedirs(f"pred/{model}", exist_ok=True)
  os.makedirs(f"submissions/{model}", exist_ok=True)

  

  test_tsd = pd.read_csv(test_file)
  # data\
  # Read spans-pred_test.txt
  pred = pd.read_csv(pred_file, sep="	", header=None)
  print("test_tsd: ", test_tsd.head())
  print("pred: ", pred.head())
  outputs = []
  for pred_3, test_tsd_3 in zip(pred.iloc[:, 1], test_tsd.iloc[:, 0]):
    # Get third row of pred, test_tsd
    # index = 3
    # pred_3 = pred.iloc[3, 1]
    pred_3 = literal_eval(pred_3)
    # Remove duplicates from list of numbers and sort
    pred_3 = list(set(pred_3))
    pred_3.sort()
    # print("pred_3: ", pred_3)
    # pred_3 = []
    # print("pred_3: ", type(pred_3))
    # test_tsd_3 = test_tsd.iloc[3, 0]
    # print("pred_3: ", pred_3)
    # print("test_tsd_3: ", test_tsd_3)
    # Get contiguous spans from list of numbers
    # https://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python
    # https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
    ranges = get_ranges(pred_3)
    # print("ranges: ", ranges)
    # Get substring from span range
    output = ""
    prev_s = 0
    for i, span in enumerate(ranges):
      if type(span) == int:
        s = span
        e = span + 1
      else:
        s = span.start
        e = span.stop + 1
      # if type(span) == int:
      #     substr = test_tsd_3[span]
      # else:
      output += test_tsd_3[prev_s:s] + "$" + test_tsd_3[s:e] + "$"
      prev_s = e
    
    if prev_s < len(test_tsd_3):
      output += test_tsd_3[prev_s:]
    
    # print("output: ", output)
    outputs.append(output)
  
  # Append outputs to test_tsd
  test_tsd["Expected"] = outputs
  print("test_tsd: ", test_tsd.head())
  test_tsd.index += 1
  print(test_tsd.head())
  # Drop the first column of test_tsd
  test_tsd = test_tsd.drop(columns=["text"])

  # Post processing
  test_tsd["Expected"] = test_tsd["Expected"].apply(replace_fixed)
  test_tsd["Expected"] = test_tsd["Expected"].apply(replace_end)

  if train_check:
    test_tsd["diff2"] = test_tsd["Expected"] != test_tsd["gt"]
    # Keep only gt, expected, diff2
    test_tsd = test_tsd[["gt", "Expected", "diff2"]]


  # Save the dataframe to a csv file start index is 1, index=True
  test_tsd.to_csv(out_file, index=True, index_label="Id")

  

  


