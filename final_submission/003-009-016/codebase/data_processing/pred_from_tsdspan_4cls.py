import pandas as pd
import os
from operator import itemgetter
from itertools import groupby
from ast import literal_eval
from pathlib import Path
import re

def get_ranges(data):
  ranges = []
  for key, group in groupby(enumerate(data), lambda x:x[0]-x[1]):
      group = list(map(itemgetter(1), group))
      if len(group) > 1:
          ranges.append(range(group[0], group[-1]))
      else:
          ranges.append(range(group[0], group[0]))
  return ranges


space_chars = [
  '।', '?', '!', ",", 
  # '"'
]
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
    # print("End with punct$", x)
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
  x =  x if x[-1] in end_chars else x + "$$"

  if x.endswith('$$ $$'):
    print("End with $$ $$", x)
    x = x[:-5] + '$$ '
  
  return x

if __name__ == "__main__":
  # Read test_tsd.csv
  test_file = "processed_data/test_tsd.csv"  

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

  # test_file = "train_3cls.csv"
  # pred_file = f"./three_class/bert_token_3cls_banglabert/spans-pred_test_checkpoint-3000.txt"
  # pred_file = f"./three_class/bert_token_3cls_bertlarge7000_banglabert3000_int.txt"
  # pred_file = f"./three_class/bert_token_3cls_bertlarge/spans-pred_test_checkpoint-7000.txt"
  # out_file = f"./three_class/bert_token_3cls_banglabert/spans-pred_train_checkpoint-2000.csv"
  # pred_file = f"./three_class/bertcrf_3cls/spans-pred-train_checkpoint-500.txt"
  # pred_file = "./pred/bert_token_4cls_bertlarge/spans-pred_test_checkpoint-8500.txt"
  # pred_file = './pred/banglabert_large_warmup0.1_lsmth0.1_4cls/spans-pred_test_checkpoint-18500.txt'

  # pred_file = "./pred/banglabert_large_warmup0.1_lsmth0.1_4cls/spans-pred_test_checkpoint-8000_sent11k.txt"
  # test_file = "../external_datasets/sazzed2019sentiment/processed/all_p8500_n3307.csv"
  # test_file = "../external_datasets/csebuetnlp_paraphrase/processed/csebuetnlp_paraphrase_5k_v2.csv"
  # test_file = "../external_datasets/facebook-scraper/BharotMaa_1000.csv"
  # pred_file = "./pred/banglabert_large_warmup0.1_lsmth0.1_4cls/spans-pred_test_checkpoint-8000_bharat1000.txt"
  # test_file = "../external_datasets/facebook-scraper/BanglaTribuneOnline_1000.csv"
  # pred_file = "./pred/banglabert_large_warmup0.1_lsmth0.1_4cls/spans-pred_test_checkpoint-8000_bangtribune.txt"
  
  # pred_file = "./pred/banglabert_large_warmup0.1_lsmth0.1_4cls/spans-pred_test_checkpoint-8000_bt_th0.8.txt"
  
  # pred_file = "./pred/banglabert_large_warmup0.1_lsmth0.1_4cls/spans-pred_test_checkpoint-18500_th0.8.txt"
  # pred_file = "./pred/banglabert_warmup0.1_lsmth0.1/spans-pred_test_checkpoint-19500.txt"
  # pred_file = './pred/ens/banglabert19500_bertlarge18500_warmup0.1_lsmth0.1_union.txt'
  # pred_file = './pred/bert_token_4cls/banglabert_large_adamwpt_warmup0.1_lsmth0.1_th0.1/spans-pred_test_checkpoint-40000.txt'
  pred_file = './pred/bert_token_4cls/banglabert_adamwpt_warmup0.1_lsmth0.1_th0.1_fp16/spans-pred_test_checkpoint-21000.txt'
  




  pred_file_p = Path(pred_file)
  out_file = pred_file_p.parent/(pred_file_p.stem + '.csv')

  # Rename predfile to csv as out_file in python
  # pred_file_p = Path(pred_file)
  # out_file = pred_file_p.parent/(pred_file_p.stem + '.csv')

  
  train_check = False
  # train_check = True
  if train_check:
    test_file = "processed_data/train_4cls.csv"
    test_file = "processed_data/test_4cls.csv"

    # pred_file = "pred/bert_large_25ep/spans-pred_train_checkpoint-8500.txt"
    # out_file = "pred/bert_large_25ep/train.csv"
    # pred_file = "pred/bertcrf/spans-pred-train.txt"
    # out_file = "pred/bertcrf/train.csv"
    # pred_file = "pred/bert/spans-pred_train_checkpoint-500.txt"
    # out_file = "pred/bert/train.csv"
    # pred_file = f"./three_class/bert_token_3cls_banglabert/spans-pred_train_checkpoint-2000.txt"
    # pred_file = f"./three_class/bertcrf_3cls/spans-pred-train_checkpoint-500.txt"
    # pred_file = "pred/bert_token_4cls_bertlarge/spans-pred_train_checkpoint-8000.txt"
    # pred_file = "pred/bert_token_4cls_bertlarge/spans-pred_validation_checkpoint-8000.txt"
    test_file = "../ninth_place_tsd/data/vasha23_4cls/test_4cls.csv"
    pred_file = "./pred/bert_token_4cls/banglabert_adamwpt_warmup0.1_lsmth0.1_th0.1_fp16/spans-pred_validation_checkpoint-16000.txt"
    
    
    pred_file_p = Path(pred_file)
    out_file = pred_file_p.parent/(pred_file_p.stem + '.csv')

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
  t_I = []
  t_E = []
  # for pred_3, test_tsd_3 in zip(pred.iloc[:, 2], test_tsd.iloc[:, 0]):
  for pred_3, pred_3_I, pred_3_E, test_tsd_3 in zip(pred.iloc[:, 1], pred.iloc[:, 2], pred.iloc[:, 3],test_tsd['text']):
    # Get third row of pred, test_tsd
    # index = 3
    # pred_3 = pred.iloc[3, 1]
    # test_tsd_3 = "লিও পেইজকে তার বাসায় আনতে চাইল, কিন্তু বাধ সাধল তার বাবা মা।"
    # pred_3 = "[52, 53, 54, 55]"
    # pred_3_I = "[57, 58]"
    pred_3 = literal_eval(pred_3)
    # Remove duplicates from list of numbers and sort
    pred_3 = list(set(pred_3))
    pred_3.sort()

    pred_3_I = literal_eval(pred_3_I)
    pred_3_I = list(set(pred_3_I))
    pred_3_I.sort()

    pred_3_E = literal_eval(pred_3_E)
    pred_3_E = list(set(pred_3_E))
    pred_3_E.sort()

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
    ranges_I = get_ranges(pred_3_I)
    ranges_E = get_ranges(pred_3_E)


    range_merged = []
    for i, rng_I in enumerate(ranges_I):
      if rng_I is None:
        continue
      for j, rng_B in enumerate(ranges):
        if rng_B is None:
          continue
        # print(rng_B, rng_I)
        if (rng_I.start == (rng_B.stop + 2)) and (rng_B.stop + 1 not in pred_3): # handle one space between them
          new_rng = range(rng_B.start, rng_I.stop)
          # print(new_rng)
          range_merged.append(new_rng)
          ranges_I[i] = None
          ranges[j] = None
          # exit()
          break
    
    # Filter all Nones from ranges and ranges_I
    ranges = [rng for rng in ranges if rng is not None]
    ranges_I = [rng for rng in ranges_I if rng is not None]

    t_I += ranges_I
    t_E += ranges_E
    ranges = ranges + ranges_I + range_merged + ranges_E
    # print("ranges: ", ranges)
    # Sort ranges by start
    ranges.sort(key=lambda x: x.start)
    
    # exit()

    # print("ranges: ", ranges)
    # Get substring from span range
    output = ""
    prev_s = 0
    
    for i, span in enumerate(ranges):
      # if type(span) == int:
      #   s = span
      #   e = span + 1
      # else:
      s = span.start
      e = span.stop + 1
      # if type(span) == int:
      #     substr = test_tsd_3[span]
      # else:
      # Forces fix for overrun
      if s >= len(test_tsd_3):
        break

      # if test_tsd_3[s:e] == "": 
      #   print(output, s, e, i, test_tsd_3, len(test_tsd_3))
      #   exit()
      #   continue
      if s in pred_3_E:
        output += test_tsd_3[prev_s:e] + "$$"
      else:
        output += test_tsd_3[prev_s:s] + "$" + test_tsd_3[s:e] + "$"

      prev_s = e
    
    if prev_s < len(test_tsd_3):
      output += test_tsd_3[prev_s:]
    
    # print("output: ", output)
    # exit()
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
    test_tsd = test_tsd[[
      "gt", 
      "Expected", 
      "diff2"]]
  

  # test_tsd = test_tsd[["gt", "Expected"]]
  # Save the dataframe to a csv file start index is 1, index=True
  test_tsd.to_csv(out_file, index=True, index_label="Id")
  # print(t_I)
  # print(t_E)
  # print(len(t_I), len(t_E))

  

  


