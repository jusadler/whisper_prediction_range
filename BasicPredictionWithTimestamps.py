import os
import time

import pandas as pd

import whisper


# TODO Rename 907 - 1058


def main():
    model_size = "large"
    prompt = ""
    os.chdir("D:/Notrufe/Pred_With_Timestamps/")
    model = whisper.load_model(model_size)
    print("Model Load Complete")
    for i in range(510, 511):
        start = time.time()
        file = f"D:/Notrufe/X_Data/{i}.wav"
        result = model.transcribe(file)
        end = time.time()
        print(f"Computing time case {i}: {end - start} seconds")
        df_list = []
        current_seek = 0
        current_text_split = ""
        last_end = 0
        for segment in result["segments"]:
            if segment.get("seek") != current_seek:
                df_list.append(
                    pd.DataFrame([[current_seek, last_end, current_text_split]], columns=["seek", "end", "text"]))
                current_seek = segment.get("seek")
                current_text_split = ""
            last_end = segment.get("end")
            current_text_split += segment.get("text")
        df_list.append(
            pd.DataFrame([[current_seek, last_end, current_text_split]], columns=["seek", "end", "text"]))

        pd.concat(df_list, ignore_index=True).to_csv(f"{i}_{model_size}_timestamps.csv")


if __name__ == '__main__':
    main()
