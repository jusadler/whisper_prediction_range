# Based on: https://stackoverflow.com/questions/42060433/python-pydub-splitting-an-audio-file # TODO Remove if works
# Based on: https://superuser.com/questions/579008/add-1-second-of-silence-to-audio-through-ffmpeg

import os

import pandas as pd

directory = "E"

y_path = f"{directory}:/Notrufe/y_data/"
os.chdir(f"{directory}:/Notrufe/X_data_split/")

for case in range(250):
    if os.path.exists(f"{y_path}{case}_timestamps.csv"):
        case_df = pd.read_csv(f"{y_path}{case}_timestamps.csv")
        # sound_file = pydub.AudioSegment.from_wav(f"{case}.wav")
        # sound_file_Value = np.array(sound_file.get_array_of_samples())
        # milliseconds in the sound track
        splits = [(row["seek"], row["end"] * 100) for index, row in case_df.iterrows()]
        for count, split in enumerate(splits):
            # new_file = sound_file_Value[10*int(split[0]):10*int(split[1])]
            # new_file = np.append(new_file, np.zeros(30000-(10*int(split[1])-10*int(split[0]))))
            # song = pydub.AudioSegment(new_file.tobytes(), frame_rate=sound_file.frame_rate,
            #                           sample_width=sound_file.sample_width, channels=1)
            # song.export(f"{case}_{count}.wav", format="wav")
            if os.path.exists(f"{directory}:/Notrufe/X_data_split/{case}_{count}.wav"):
                os.remove(f"{directory}:/Notrufe/X_data_split/{case}_{count}.wav")
            if os.path.exists(f"{directory}:/Notrufe/X_data_split/{case}_{count}_unpadded.wav"):
                os.remove(f"{directory}:/Notrufe/X_data_split/{case}_{count}_unpadded.wav")
            start = round(split[0] / 100, 2)
            end = round(split[1] / 100, 2)
            os.system(
                f"ffmpeg -i {directory}:/Notrufe/X_data_split/{case}.wav -ss {start} -to {end} {directory}:/Notrufe/X_data_split/{case}_{count}_unpadded.wav")
            # if end-start < 30:
            # os.system(f'ffmpeg -i D:/Notrufe/X_data_split/{case}_{count}_unpadded.wav -af "apad=pad_dur={30+start-end}" D:/Notrufe/X_data_split/{case}_{count}.wav')
            os.system(
                f'ffmpeg -i {directory}:/Notrufe/X_data_split/{case}_{count}_unpadded.wav -af "apad=whole_dur=30" {directory}:/Notrufe/X_data_split/{case}_{count}.wav')

            # else:
            #    os.system(f"cp D:/Notrufe/X_data_split/{case}_{count}_unpadded.wav D:/Notrufe/X_data_split/{case}_{count}.wav")
