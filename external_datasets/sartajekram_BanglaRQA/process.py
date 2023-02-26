import pandas as pd
import pdb

train_file = 'gt/Train.json'
output_file = 'processed/sartajekram_BanglaRQA_train.csv'

df_train = pd.read_json(train_file)
print(df_train.head())
print(df_train['data'][0])
# pdb.set_trace()

# Extract all the questions and answers from the train file
questions = []
passage_ids = []

# Iterate rows of the dataframe
for index, row in df_train.iterrows():
  for qa in row['data']['qas']:
    questions.append(qa['question_text'])
    passage_ids.append(index)

# Create a dataframe with the questions and passage ids
df_questions = pd.DataFrame({'question': questions, 'passage_id': passage_ids})
print(df_questions.head(), df_questions.shape)

# Rename question column to text
df_questions.rename(columns={'question': 'text'}, inplace=True)

# Save
df_questions.to_csv(output_file, index=False)

# Sample a 5k subset of the train file
df_questions = df_questions.sample(n=5000, random_state=42)
df_questions.to_csv('processed/sartajekram_BanglaRQA_train_5k.csv', index=False)


