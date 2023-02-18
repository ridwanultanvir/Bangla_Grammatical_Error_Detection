import pandas as pd
import re
from tqdm import tqdm
tqdm.pandas()

def get_spans(sentence):
  # sentence = "যানি না এই কমেন্ট পড়বেন কিনা ।,$যানি না$ এই কমেন্ট পড়বেন কিনা$ $।"
  # sentence = "$যানি না$ এই কমেন্ট পড়বেন কিনা$ $।"
  # sentence = "মা যদি আমার জীবনচরিত লেখেন$$ $তা হলে$ আমি সকাল সকাল মরতে রাজি আছি।"
  # sentence = "a$bc$bcde$efg$gh$$ij$dgd$"
  # print(sentence)
  # Find all the span indices between $ and $ without considering the $ signs
  span_indices = [m.span() for m in re.finditer(r"\$.*?\$", sentence)]
  # Don't touch the sentence if the number of dollars is odd
  # Remove all the $ signs from the sentence
  out_sentence = sentence.replace("$", "")
  # Check whether the out_sentence contains punctuation marks
  # print(span_indices)
  spans = []
  count_ignore = 0
  # B = 1, I = 2
  Bs = []
  Is = []
  for i, span in enumerate(span_indices):
    if span[1] - span[0] < 3: # ignore $$ for now
      count_ignore += 1
      continue
    offset = 2*i + 1
    start = span[0] + 1
    start = start - offset
    end = span[1] - 1
    end = end - offset

    # Add to I after first zero is found in out_sentence[start:end]
    mid = 0
    chunk = out_sentence[start:end]
    # print("chunk: ", chunk)
    if len(chunk) == 0: continue
    # print("chunk[0]: ", out_sentence[start])

    # if chunk[0] == " ":
    Bs.append(start)
    # print("Added to Bs: ", start)
    
    for j in range(start + 1, end):
      if out_sentence[j] == " ":
        mid = 1
      if mid: Is.append(j)
      else: Bs.append(j)


    # print("start: ", start, "end: ", end, "offset: ", offset)
    output_span = list(range(start, end))
    # spans.append(output_span)
    spans.extend(output_span)
  
  # print("Output sentence: ", out_sentence)
  # print("Output spans: ", spans)
  # print("Bs: ", Bs)
  # print("Is: ", Is)
  # print("Number of ignored spans: ", count_ignore)
  if count_ignore > 0:
    with open("ignored_3cls.txt", "a", encoding="utf-8") as f:
      f.write(sentence + "\n")
    # print("Sentence: ", sentence)
  # exit()
  return out_sentence, spans, count_ignore, Bs, Is

if __name__ == "__main__":
  # Read data from DataSetFold1.csv\DataSetFold1.csv
  import pandas as pd
  df = pd.read_csv(r"DataSetFold1.csv/DataSetFold1.csv")
  # Apply the get_spans function to the text column
  df["text"], df["spans"], df["count_ignore"], df["Bs"], df["Is"] = zip(*df["gt"].apply(get_spans))
  print("Sum of ignored spans: ", df["count_ignore"].sum())
  # exit()
  # Check whether gt and text columns are equal
  df["same"] = df["gt"] == df["text"]
  # Find the levenstein distance between gt and text columns
  from nltk.metrics import edit_distance
  tqdm.pandas(desc="Calculating edit distance")
  df["distance"] = df.progress_apply(lambda x: edit_distance(x["gt"], x["text"]), axis=1)
  # Save the dataframe to a csv file
  stat = df['same'].value_counts()
  print(stat)
  df.to_csv("DataSetFold1_processed_3cls.csv", index=False)
  #Create .2 train test split
  from sklearn.model_selection import train_test_split
  train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['same'])
  train.to_csv("train_3cls.csv", index=False)
  test.to_csv("test_3cls.csv", index=False)


  

    





