import os
import time

import whisper
import torch


# TODO Rename 907 - 1058


def main():
    model_size = "large"
    prompt = ""
    os.chdir("E:/Modelle/training_test/try_new_save/")
    # model = whisper.load_model("E:/Modelle/training_test/try_new_save/save_dict/state_dict.pth")
    model = torch.load("E:/Modelle/training_test/try_new_save/model.pth")
    print("Model Load Complete")
    for i in range(0, 500):
        start = time.time()
        file = f"E:/Notrufe/X_Data/{i}.wav"
        result = model.transcribe(file, border=0)
        end = time.time()
        print(f"Computing time case {i}: {end - start} seconds")
        try:
            with open(f"Predictions/{i}_{model_size}_first_test_run.txt", "w") as text_file:
                text_file.write(result["text"])
        except Exception as e:
            print(e)
            print(result)


if __name__ == '__main__':
    main()
