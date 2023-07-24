
# Based on https://www.youtube.com/watch?v=O60EnXcbi6g

from dataclasses import dataclass
from datasets import Dataset, Audio
from typing import Any, Dict, List, Union
from os.path import exists
import torch
import pandas as pd
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
model_path = "C:\\Users\\Admin\\.cache\\whisper\\large-v2.pt"  # ("E:\Modelle\large-v2.pt"
# Load Feature extractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")

# Load Tokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", language="German", task="transcribe")

# Combine extractor and tokenizer
processor = WhisperProcessor.from_pretrained("openai/whisper-large", language="German", task="transcribe")


def create_dataset():
    first_case = 0
    last_case = 400
    X_path = "D:/Notrufe/X_Data/"
    y_path = "D:/Notrufe/y_data/"
    existing_ground_truth_cases = [i for i in range(first_case, last_case) if exists(f'{y_path}{i}.txt')]
    # dataset = Dataset.from_dict(
    #     {"audio": [f'{X_path}{i}.wav' for i in existing_ground_truth_cases],
    #      "transcription": [f'{y_path}{i}.txt' for i in existing_ground_truth_cases]}
    # ).cast_column("audio", Audio())
    dataset = Dataset.from_dict(
        {"audio": [f'{X_path}{i}.wav' for i in existing_ground_truth_cases]}
    ).cast_column("audio", Audio())
    for index, case in enumerate(existing_ground_truth_cases):
        with open(f"{y_path}{case}.txt", "r") as file:
            transcription = file.read().replace('\n', '')
            # dataset[index]["audio"] = dataset[index]["audio"].update([('transcription', transcription)])
            dataset[index]["audio"]["transcription"] = transcription
    print(dataset[0])


create_dataset()
