if __name__ == "__main__":
  import pandas as pd
  df = pd.read_csv(r"DataSetFold1.csv/DataSetFold1.csv")
  # Concat all the string into one in the text column
  corpus = df["sentence"].str.cat(sep=" ")
  print(len(corpus))
  # from bnlp.corpus import punctuations
  punctuations = "!#$%&'()*+,-./:;<=>?@[\]^_`{|}~।ঃ"
  punctuations = list(x for x in punctuations)
  punctuations += ['']
  # Find the frequency of each punctuation in the corpus
  stat = pd.Series(corpus).apply(lambda x: pd.value_counts(list(x))).sum(axis = 0)
  # Filter out the punctuations from the stat
  print(punctuations)
  stat = stat[stat.index.isin(list(x for x in punctuations))]
  # Sort the stat in descending order
  stat = stat.sort_values(ascending=False)
  print(stat)
  important_punctuations = ['।', '?', '!', ","]