import pandas as pd
from tqdm import tqdm
import re
import pdb
from collections import defaultdict

from ner import NER

model_path = "model/bn_ner.pkl"
bn_ner = NER(model_path=model_path)

trainfile = r"../data/DataSetFold1_u.csv/DataSetFold1_u.csv"
col = "sentence"
trainfile = r"../data/submissions/test_results_bn3500_bnlargefull1_int.csv"
col = "Expected"

errorfile = r"data/bad_words.csv"

train = pd.read_csv(trainfile, encoding='utf-8')
sentences = train[[col]]
errors = pd.read_csv(errorfile, encoding='utf-8')

error_regex = "|".join(errors.values.flatten().tolist())
errors_list = errors.values.flatten().tolist()
# Check if any of the words in the sentence are in the dictionary
file = open("data/trainerrors.txt", "w", encoding="utf-8")
is_train = "DataSetFold" in trainfile
sentence_list = sentences[col].tolist()

for idx, sentence in tqdm(enumerate(sentence_list), total=len(sentence_list)):
  # for err in errors.values:
  # Find all the errors in the sentence
  # err = re.findall(error_regex, sentence)
  ner_tags = bn_ner.tag(sentence)
  # ner_dict = defaultdict(str, "O")
  # Create an defaultdict of string with default value "O"
  ner_dict = defaultdict(lambda: "O")
  for tag in ner_tags:
    ner_dict[tag[0]] = tag[1]

  for tid, token in enumerate(sentence.split()):
    if ner_dict[token] != "O":
      continue
    # try:
    #   if ner_tags[tid][1] != "O":
    #     print(token, "NER:", ner_tags[tid][1])
    #     continue
    # except:
    #   pdb.set_trace()
    
    if re.fullmatch(error_regex, token, ):
      if is_train:
        file.write(token + ", " + train["gt"][idx] + "\n")
      else:
        file.write(token + ", " + sentence + "\n")
      file.flush()
      break
    # if token in errors_list:
    # for err in errors_list:
    #   if token == err:
    #     # print(token, err); exit()
    #     file.write(token + ", " + err + ", " + sentence + "\n")
    #     file.flush()
    #     break

  # if err:
  #   # print(sentence)
  #   # pdb.set_trace()
  #   file.write(err[0] + ", " + sentence + "\n")
  #   file.flush()
  #   # break

file.close()

# if err in sentence:
#   print(sentence)
#   break
  

