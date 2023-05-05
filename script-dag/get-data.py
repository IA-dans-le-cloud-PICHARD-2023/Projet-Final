import glob
import os
import pandas as pd

# list all csv files only
csv_files = glob.glob('./kaggle/input/dataset/*.{}'.format('csv'))
print(csv_files)

# append all the CSV files into one dataframe
# concatenate all the CSV files into one dataframe
df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

df_csv_concat.to_csv('./outputs/datas/combined_dataset.csv', index=False)