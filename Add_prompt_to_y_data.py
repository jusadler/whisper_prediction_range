from os.path import exists

import pandas as pd
import os

os.chdir("E:/Notrufe/y_data/")
for i in range(511):
    if exists(f'{i}.txt'):
        y_df = pd.read_csv(f"{i}_large_timestamps.csv")
        prompt_list = [""]
        for index, row in y_df.iterrows():
            prompt = prompt_list[-1] if prompt_list else ""
            prompt_list.append(prompt + " " + str(row["text"]))
        prompt_list.pop()
        y_df["prompt"] = prompt_list
        y_df.to_csv(f"{i}_timestamps.csv")
    else:
        print(i)
