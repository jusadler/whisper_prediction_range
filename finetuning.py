# Based on https://huggingface.co/blog/fine-tune-whisper

import evaluate
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer

from DataCollator import DataCollatorEmergencyCalls
from EmergencyCallsDataset import EmergencyCallsDataset
from EmergencyCallsValidationDataset import EmergencyCallsValidationDataset


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


emergency_calls = EmergencyCallsDataset()

model_path = "E:/Modelle/large-v2.pt"  # ("E:\Modelle\large-v2.pt" "C:\\Users\\Admin\\.cache\\whisper\\large-v2.pt
# Load Feature extractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")

# Load Tokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", language="German", task="transcribe")

# Combine extractor and tokenizer
processor = WhisperProcessor.from_pretrained("openai/whisper-large", language="German", task="transcribe")

# Check if feature extractor and tokenizer work as expected
# _, input_str, _ = emergency_calls[0]
# labels = tokenizer(input_str).input_ids
# decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
# decoded_str = tokenizer.decode(labels, skip_special_tokens=True)
#
# print(f"Input:                 {input_str}")
# print(f"Decoded w/ special:    {decoded_with_special}")
# print(f"Decoded w/out special: {decoded_str}")
# print(f"Are equal:             {input_str == decoded_str}")


# def prepare_dataset(batch):
#     signal, transcript, sampling_rate = batch["audio"]
#     batch["input_features"] = feature_extractor(signal, sampling_rate=sampling_rate).input_features[0]
#     batch["labels"] = tokenizer(batch["sentence"]).input_ids
#
#
# emergency_calls = emergency_calls.map(prepare_dataset, num_proc=1)
data_collator = DataCollatorEmergencyCalls(processor=processor)
metric = evaluate.load("wer")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
training_args = Seq2SeqTrainingArguments(
    output_dir="E:/Modelle/training_test/try_large_run",  # change to a repo name of your choice
    per_device_train_batch_size=16,  # TODO BACK TO 16!!
    # per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    # warmup_steps=125,  # back to 500
    # max_steps=1000,  # back to 4000
    num_train_epochs=1,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=250,
    logging_steps=250,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=EmergencyCallsDataset(),
    eval_dataset=EmergencyCallsValidationDataset(),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()

trainer.save_model("E:/Modelle/training_test/try_new_save")
torch.save(trainer.model.state_dict(), "E:/Modelle/training_test/try_new_save/save_dict/state_dict.pth")
torch.save(trainer.model, "E:/Modelle/training_test/try_new_save/model.pth")
