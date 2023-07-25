import pandas as pd
from os.path import exists


def create_metadata_file():
    first_case = 0
    last_case = 400
    X_path = "D:/Notrufe/X_Data/"
    y_path = "D:/Notrufe/y_data/"
    existing_ground_truth_cases = [i for i in range(first_case, last_case) if exists(f'{y_path}{i}.txt')]
    metadate_list = []
    for case in existing_ground_truth_cases:
        with open(f"{y_path}{case}.txt", "r") as file:
            transcription = file.read().replace('\n', '')
            metadate_list.append({'FilePath': f'{X_path}{case}.wav', 'Transcription': transcription})
    metadata_df = pd.DataFrame(metadate_list, columns=["FilePath", "Transcription"])
    metadata_df.to_csv("D:/Notrufe/metadata.csv")
