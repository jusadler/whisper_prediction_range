from os.path import exists

import pandas as pd


def create_metadata_file():
    directory = "E"
    first_case = 100
    last_case = 124
    X_path = f"{directory}:/Notrufe/X_Data_Split/"
    y_path = f"{directory}:/Notrufe/y_data/"
    existing_ground_truth_cases = [i for i in range(first_case, last_case + 1) if exists(f'{y_path}{i}.txt')]
    metadate_list = []
    for case in existing_ground_truth_cases:
        case_df = pd.read_csv(f"{y_path}{case}_timestamps.csv")
        for index, row in case_df.iterrows():
            metadate_list.append(
                {'FilePath': f'{X_path}{case}_{index}.wav', 'Transcription': row["text"], 'Prompt': row["prompt"]})
    metadata_df = pd.DataFrame(metadate_list, columns=["FilePath", "Transcription", "Prompt"])
    metadata_df.to_csv(f"{directory}:/Notrufe/metadata_split_validation.csv")


create_metadata_file()
