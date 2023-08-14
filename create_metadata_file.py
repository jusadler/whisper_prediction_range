from os.path import exists

import pandas as pd


def create_metadata_file():
    split = False
    directory = "E"
    first_case = 201
    last_case = 249  # TODO Regenerate
    X_path = f"{directory}:/Notrufe/X_Data_Split/"
    y_path = f"{directory}:/Notrufe/y_data/"
    existing_ground_truth_cases = [i for i in range(first_case, last_case + 1) if exists(f'{y_path}{i}.txt')]
    metadata_list = []
    if split:
        for case in existing_ground_truth_cases:
            case_df = pd.read_csv(f"{y_path}{case}_timestamps.csv")
            for index, row in case_df.iterrows():
                metadata_list.append(
                    {'FilePath': f'{X_path}{case}_{index}.wav', 'Transcription': row["text"], 'Prompt': row["prompt"]})
        metadata_df = pd.DataFrame(metadata_list, columns=["FilePath", "Transcription", "Prompt"])
        metadata_df.to_csv(f"{directory}:/Notrufe/metadata_split_validation.csv")
    else:
        for case in existing_ground_truth_cases:
            with open(f"{y_path}{case}.txt", "r") as text_file:
                transcript = text_file.read().replace('\n', '')
            metadata_list.append({'FilePath': f'{X_path}{case}.wav', 'Transcription': {transcript}})
        metadata_df = pd.DataFrame(metadata_list, columns=["FilePath", "Transcription"])
        metadata_df.to_csv(f'{directory}:/Notrufe/metadata_validation.csv')


create_metadata_file()
