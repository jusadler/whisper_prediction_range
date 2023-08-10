import os
import time

import whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import pipeline
from whisper import transcribe


def main():
    # pipe = pipeline(task="automatic-speech-recognition", model="E:/Modelle/training_test/try_new_save/", device="gpu",
    #                generate_kwargs={"language": "<|de|>", "task": "transcribe"})

    os.chdir("E:/Modelle/training_test/try_new_save/")
    # model = whisper.load_model("E:/Modelle/training_test/try_new_save/save_dict/state_dict.pth")
    model = whisper.load_model("E:/Modelle/large-v2.pt")
    # device = "cuda"
    # model = WhisperForConditionalGeneration.from_pretrained("E:/Modelle/training_test/try_new_save/").to(device)
    # processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    # inputs = processor.feature_extractor("E:/Notrufe/X_Data/0.wav", return_tensors="pt").input_features.to(device)
    # generate_ids = model.generate(inputs, max_length=480000, language="<|de|>", task="transcribe")
    # results = processor.tokenizer.decode(generate_ids[0])
    print("Model Load Complete")
    for i in range(0, 500):

        start = time.time()
        file = f"E:/Notrufe/X_Data/{i}.wav"
        # result = pipe(file)
        result = transcribe(model=model, audio=file, language="de")
        end = time.time()
        print(f"Computing time case {i}: {end - start} seconds")
        try:
            with open(f"Predictions/{i}_first_test_run.txt", "w") as text_file:
                text_file.write(result["text"])
        except Exception as e:
            print(e)
            print(result)


if __name__ == '__main__':
    main()
