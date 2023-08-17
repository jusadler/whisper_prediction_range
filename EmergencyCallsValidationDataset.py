import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

import whisper.tokenizer
from whisper import log_mel_spectrogram
from whisper.audio import N_SAMPLES


# Based on https://www.youtube.com/watch?v=88FFnqt5MNI


class EmergencyCallsValidationDataset(Dataset):

    def __init__(self, annotations_path, path_only=False, data_to_gpu=True):
        self.annotations = pd.read_csv(annotations_path, index_col=0)
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language="de", task="transcription")
        self.path_only = path_only
        self.data_to_gpu = data_to_gpu

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self.annotations.iloc[index, 0]
        if self.path_only:
            return audio_sample_path, self.annotations.iloc[index, 1]
        transcription = torch.tensor(self.tokenizer.encode(self.annotations.iloc[index, 1]))
        # prompt = self.annotations.iloc[index, 2]
        # if np.isnan(prompt):
        #     prompt = ""
        # signal, _ = torchaudio.load(audio_sample_path)
        signal = log_mel_spectrogram(audio_sample_path)
        if self.data_to_gpu:
            transcription.to(0)
            signal.to(0)
        # return signal, transcription, prompt
        return signal, transcription

    def __getitems__(self, indices):
        return_list = []
        for index in indices:
            return_list.append(self.__getitem__(index))
        return return_list


if __name__ == "__main__":

    emergency_call_dataset = EmergencyCallsValidationDataset()

    print(f"There are {len(emergency_call_dataset)} samples in the dataset")

    signal_1, transcription_1, _ = emergency_call_dataset[1]

    a = 1
