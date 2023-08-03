import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

from whisper import log_mel_spectrogram
from whisper.audio import N_SAMPLES


# Based on https://www.youtube.com/watch?v=88FFnqt5MNI


class EmergencyCallsValidationDataset(Dataset):

    def __init__(self):
        self.annotations = pd.read_csv("E:/Notrufe/metadata_split_validation.csv", index_col=0)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self.annotations.iloc[index, 0]
        transcription = self.annotations.iloc[index, 1]
        # prompt = self.annotations.iloc[index, 2]
        # if np.isnan(prompt):
        #     prompt = ""
        # signal, _ = torchaudio.load(audio_sample_path)
        signal = log_mel_spectrogram(audio_sample_path)
        # return signal, transcription, prompt
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
