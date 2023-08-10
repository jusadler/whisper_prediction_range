import os

import pandas as pd
import pydub

y_path = "D:/Notrufe/y_data/"
os.chdir("D:/Notrufe/X_data_split/")

for case in range(125):
    if os.path.exists(f"{y_path}{case}_timestamps.csv"):
        case_df = pd.read_csv(f"{y_path}{case}_timestamps.csv")
        # sound_file = pydub.AudioSegment.from_wav(f"{case}.wav")
        # sound_file_Value = np.array(sound_file.get_array_of_samples())
        # milliseconds in the sound track
        splits = [(row["seek"], row["end"] * 100) for index, row in case_df.iterrows()]
        for count, split in enumerate(splits):
            sound_file = pydub.AudioSegment.from_wav(f"{case}_{count}.wav")
            if(len(sound_file) != 30000):
                unpadded_sound_file = pydub.AudioSegment.from_wav(f"{case}_{count}_unpadded.wav")
                print(f"Case: {case}, Count: {count}, Length: {len(sound_file)}, Unpadded Length: {len(unpadded_sound_file)}")
            # new_file = sound_file_Value[10*int(split[0]):10*int(split[1])]
            # new_file = np.append(new_file, np.zeros(30000-(10*int(split[1])-10*int(split[0]))))
            # song = pydub.AudioSegment(new_file.tobytes(), frame_rate=sound_file.frame_rate,
            #                           sample_width=sound_file.sample_width, channels=1)
            # song.export(f"{case}_{count}.wav", format="wav")
