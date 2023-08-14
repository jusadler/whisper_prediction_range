import os
import time

import whisper
import torch


# TODO Rename 907 - 1058


def main():
    model_size = "small"
    model_name = "model_checkpoint_20230811_113629_4"
    prompt = ""
    os.chdir("E:/Modelle/training_test/v2_test_1")
    model = whisper.load_model(f"{model_name}.pt", local_model=True)
    # model = torch.load(f"{model}.pt")
    model.eval()
    print("Model Load Complete")
    os.chdir("E:/Modelle/training_test/v2_test_1/Predictions")
    for i in range(0, 100):
        start = time.time()
        file = f"E:/Notrufe/X_Data/{i}.wav"
        result = model.transcribe(file, language="de")
        end = time.time()
        print(f"Computing time case {i}: {end - start} seconds")
        try:
            with open(f"{i}_{model_size}_{model_name}.txt", "w") as text_file:
                text_file.write(result["text"])
        except Exception as e:
            print(e)
            print(result)


if __name__ == '__main__':
    main()
