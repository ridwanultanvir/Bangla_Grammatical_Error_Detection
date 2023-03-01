import pandas as pd

# Load the en_bn.csv file
df_en_bn = pd.read_csv('en_bn.csv')

# Remove rows where the bangla column has only one character
df_en_bn = df_en_bn[df_en_bn['bangla'].apply(lambda x: len(x) > 1)]

# Save the modified dataframe to a new CSV file
df_en_bn.to_csv('new.csv', index=False)
