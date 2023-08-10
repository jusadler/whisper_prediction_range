import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torchaudio

import whisper.tokenizer
from whisper import log_mel_spectrogram
from whisper.audio import N_SAMPLES


# Based on https://www.youtube.com/watch?v=88FFnqt5MNI


class EmergencyCallsDataset(Dataset):

    def __init__(self):
        self.annotations = pd.read_csv("E:/Notrufe/metadata_split.csv", index_col=0)
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language="de", task="transcribe")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self.annotations.iloc[index, 0]
        transcription = self.annotations.iloc[index, 1]
        prompt = self.annotations.iloc[index, 2]
        if not isinstance(prompt, str):
            prompt = ""
        # signal, _ = torchaudio.load(audio_sample_path)
        if not isinstance(transcription, str):
            transcription = ""
        transcription = torch.tensor(self.tokenizer.encode(transcription)).to(0)
        signal = log_mel_spectrogram(audio_sample_path).to(0)
        # return signal, transcription, transcription  # TODO CHANGE BACK TO PROMPT ON 3rd POSITION
        return signal, transcription

    def __getitems__(self, indices):
        return_list = []
        for index in indices:
            return_list.append(self.__getitem__(index))
        return return_list


if __name__ == "__main__":

    emergency_call_dataset = EmergencyCallsDataset()

    print(f"There are {len(emergency_call_dataset)} samples in the dataset")

    signal_1, transcription_1, _ = emergency_call_dataset[1]

    a = 1
