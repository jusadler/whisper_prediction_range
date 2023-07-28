import os
import time

import whisper


# TODO Rename 907 - 1058


def main():
    model_size = "large"
    prompt = ""
    os.chdir("D:/Notrufe/Left_Padding_Test/")
    model = whisper.load_model(model_size)
    print("Model Load Complete")
    for i in range(0, 100):
        start = time.time()
        file = f"D:/Notrufe/X_Data/{i}.wav"
        result = model.transcribe(file, border=3)
        end = time.time()
        print(f"Computing time case {i}: {end - start} seconds")
        try:
            with open(f"{i}_{model_size}_left_padding_3_test.txt", "w") as text_file:
                text_file.write(result["text"])
        except Exception as e:
            print(e)
            print(result)


if __name__ == '__main__':
    main()
