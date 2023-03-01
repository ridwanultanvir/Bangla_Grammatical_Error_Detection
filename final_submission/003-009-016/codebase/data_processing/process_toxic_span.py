import pandas as pd
import re
from tqdm import tqdm
tqdm.pandas()

def get_spans(sentence):
  # sentence = "যানি না এই কমেন্ট পড়বেন কিনা ।,$যানি না$ এই কমেন্ট পড়বেন কিনা$ $।"
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
  for i, span in enumerate(span_indices):
    if span[1] - span[0] < 3: # ignore $$ for now
      count_ignore += 1
      continue
    start = span[0] + 1
    end = span[1] - 1
    offset = 2*i + 1
    # print("start: ", start, "end: ", end, "offset: ", offset)
    output_span = list(range(start - offset, end - offset))
    # spans.append(output_span)
    spans.extend(output_span)
  
  # print("Output sentence: ", out_sentence)
  # print("Output spans: ", spans)
  # print("Number of ignored spans: ", count_ignore)
  if count_ignore > 0:
    with open("processed_data/ignored.txt", "a", encoding="utf-8") as f:
      f.write(sentence + "\n")
    # print("Sentence: ", sentence)
  return out_sentence, spans, count_ignore

if __name__ == "__main__":
  # Read data from DataSetFold1.csv\DataSetFold1.csv
  import pandas as pd
  df = pd.read_csv(r"DataSetFold1_u.csv/DataSetFold1_u.csv")
  df2 = pd.read_csv(r"DataSetFold2.csv/DataSetFold2.csv")
  df = df.append(df2)
  # print(df.shape); exit()

  # Apply the get_spans function to the text column
  df["text"], df["spans"], df["count_ignore"] = zip(*df["gt"].apply(get_spans))
  print("Sum of ignored spans: ", df["count_ignore"].sum())
  # Check whether gt and text columns are equal
  df["same"] = df["gt"] == df["text"]
  # Find the levenstein distance between gt and text columns
  from nltk.metrics import edit_distance
  tqdm.pandas(desc="Calculating edit distance")
  df["distance"] = df.progress_apply(lambda x: edit_distance(x["gt"], x["text"]), axis=1)
  # Save the dataframe to a csv file
  stat = df['same'].value_counts()
  print(stat)
  df.to_csv("processed_data/DataSetFold1_processed.csv", index=False)
  #Create .2 train test split
  from sklearn.model_selection import train_test_split
  # train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['same'])
  # print(df['distance'].value_counts())
  # import pdb; pdb.set_trace()

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

  train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['distance'], )
  
  train.to_csv("processed_data/train.csv", index=False)
  test.to_csv("processed_data/valid.csv", index=False)


  

    





