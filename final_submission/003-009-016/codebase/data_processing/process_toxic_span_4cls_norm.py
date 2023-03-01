import pandas as pd
import re
from tqdm import tqdm
from normalizer import normalize
tqdm.pandas()

def get_spans(sentence):
  # sentence = "যানি না এই কমেন্ট পড়বেন কিনা ।,$যানি না$ এই কমেন্ট পড়বেন কিনা$ $।"
  # sentence = "$যানি না$ এই কমেন্ট পড়বেন কিনা$ $।"
  # sentence = "মা যদি আমার জীবনচরিত  লেখেন$$ $তা হলে$ আমি সকাল সকাল মরতে রাজি আছি।"
  # sentence = "লেখেন$$ $তা হলে$ আমি সকাল সকাল মরতে রাজি আছি।"
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
  Bs = [] # Begin
  Is = [] # Inside
  Es = [] # $$ handle
  for i, span in enumerate(span_indices):
    offset = 2*i + 1
    start = span[0] + 1
    start = start - offset
    end = span[1] - 1
    end = end - offset

    if span[1] - span[0] < 3: # ignore $$ for now
      count_ignore += 1
      # continue
      print("Output sentence: ", out_sentence)
      print("Ignored span: ", span)
      # Find the index of the first space before the span from end
      print("out_sentence[:start]: ", out_sentence[:start].replace(" ", "X"))
      index = out_sentence[:start].rfind(" ")
      index += 1
      print("index: ", index, "out_sentence[index]: ", out_sentence[:index].replace(" ", "X"))
      print("span:", out_sentence[index:start].replace(" ", "X"))
      
      Es+= list(range(index, start))
      print("Added to Es: ", range(index, start))
      # exit()
      continue
  
    # offset = 2*i + 1
    # start = span[0] + 1
    # start = start - offset
    # end = span[1] - 1
    # end = end - offset

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
    # print(out_sentence[start:end].replace(" ", "X"))
    # spans.append(output_span)
    spans.extend(output_span)
  
  # print("Output sentence: ", out_sentence)
  # print("Output spans: ", spans)
  # print("Bs: ", Bs)
  # print("Is: ", Is)
  # print("Number of ignored spans: ", count_ignore)
  if count_ignore > 0:
    with open("processed_data/ignored_4cls_norm.txt", "a", encoding="utf-8") as f:
      f.write(sentence + "\n")
    # print("Sentence: ", sentence)
  # exit()
  return out_sentence, spans, count_ignore, Bs, Is, Es

if __name__ == "__main__":
  with open("processed_data/ignored_4cls_norm.txt", "w", encoding="utf-8") as f:
    f.write("")
  # Read data from DataSetFold1.csv\DataSetFold1.csv
  import pandas as pd
  df = pd.read_csv(r"DataSetFold1_u.csv/DataSetFold1_u.csv")
  df2 = pd.read_csv(r"DataSetFold2.csv/DataSetFold2.csv")
  df = df.append(df2)
  # Apply the normalize function to the sentence, gt columns
  from tqdm import tqdm
  tqdm.pandas(desc="Normalizing sentence")
  df["sentence"] = df["sentence"].progress_apply(normalize)
  tqdm.pandas(desc="Normalizing gt")
  df["gt"] = df["gt"].progress_apply(normalize)
  # Apply the get_spans function to the text column
  df["text"], df["spans"], df["count_ignore"], df["Bs"], df["Is"], df["Es"] = zip(*df["gt"].apply(get_spans))
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
  df.to_csv("processed_data/DataSetFold1_processed_4cls_norm.csv", index=False)
  #Create .2 train test split
  from sklearn.model_selection import train_test_split

  # Find the levenstein distances whose count is 1
  distances = df['distance'].value_counts()
  print(distances)
  distances = distances[distances == 1]
  print(distances)
  # # Find the indices of the rows with the above distances, and replace them with next higher distance which has count > 1
  # for distance in distances.index:
  #   df.loc[df['distance'] == distance, 'distance'] = distance + 1

  for distance in distances.index:
    df.loc[df['distance'] == distance, 'distance'] = 0
  
  distances = df['distance'].value_counts()
  print(distances)

  train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['distance'])
  print(train['distance'].value_counts())
  print(test['distance'].value_counts())

  train.to_csv("processed_data/train_4cls_norm.csv", index=False)
  test.to_csv("processed_data/test_4cls_norm.csv", index=False)


  

    





