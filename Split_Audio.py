# Based on: https://stackoverflow.com/questions/42060433/python-pydub-splitting-an-audio-file

import os

import numpy as np
import pandas as pd
import pydub

y_path = "D:/Notrufe/y_data/"
os.chdir("D:/Notrufe/X_data_split/")

for case in range(50):
    case_df = pd.read_csv(f"{y_path}{case}_timestamps.csv")
    sound_file = pydub.AudioSegment.from_wav(f"{case}.wav")
    sound_file_Value = np.array(sound_file.get_array_of_samples())
    # milliseconds in the sound track
    ranges = [(case_df["seek"], case_df["end"] * 100) for index, row in case_df.itterrows()]
    # TODO Try to use enumerate
    counter = 0
    for x, y in ranges:
        new_file = sound_file_Value[x: y]
        song = pydub.AudioSegment(new_file.tobytes(), frame_rate=sound_file.frame_rate,
                                  sample_width=sound_file.sample_width, channels=1)
        song.export(f"{case}_{counter}.wav", format="wav")
        counter += 1
