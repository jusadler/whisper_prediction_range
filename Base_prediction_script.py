import os
import time

import whisper
import torch


# TODO Rename 907 - 1058
# TODO Rerun those with failed save (18, 133, 164)

def main():
    model_size = "large-v1"
    # model_name = "_model_checkpoint_20230818_140128_2"
    model_name = ""
    prompt = ""
    os.chdir("E:/Modelle/training_test/tiny_long_training_batch16")
    # model = whisper.load_model(f"{model_name}.pt", local_model=True)
    model = whisper.load_model(model_size)
    model.eval()
    print("Model Load Complete")
    os.chdir("E:/Notrufe/Large_V1_Predictions")
    for i in range(0, 511):
        start = time.time()
        file = f"E:/Notrufe/X_Data/{i}.wav"
        result = model.transcribe(file, language="de")  # , temperature=0)
        end = time.time()
        print(f"Computing time case {i}: {end - start} seconds")
        try:
            with open(f"{i}_{model_size}{model_name}.txt", "w") as text_file:
                text_file.write(result["text"])
        except Exception as e:
            print(e)
            print(result)


if __name__ == '__main__':
    main()
