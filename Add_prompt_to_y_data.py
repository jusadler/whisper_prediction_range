import pandas as pd
import os

os.chdir("D:/Notrufe/y_data/")
for i in range(50):
    y_df = pd.read_csv(f"{i}_large_timestamps.csv")
    prompt_list = [""]
    for index, row in y_df.iterrows():
        prompt = prompt_list[-1] if prompt_list else ""
        prompt_list.append(prompt + " " + row["text"])
    prompt_list.pop()
    y_df["prompt"] = prompt_list
    y_df.to_csv(f"{i}_timestamps.csv")
