import pandas as pd

# Read the CSV file into a dataframe
df = pd.read_csv('./homonyms.csv')

# Extract the first element from each array in the homonyms column
df['homonyms'] = df['homonyms'].apply(lambda x: x.strip("[]").split(",")[0].strip("'"))

# Display the resulting dataframe
df.to_csv('./homonyms2.csv', index=False)
