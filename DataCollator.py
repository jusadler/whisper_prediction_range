# Based on https://huggingface.co/blog/fine-tune-whisper
from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class DataCollatorEmergencyCalls:
    processor: Any

    def __call__(self, features) -> Dict[str, torch.tensor]:
        # dataset = EmergencyCallsDataset()
        # dataset_list = [dataset[i] for i in range(len(dataset))]
        # input_features = [{"input_features": dataset_list[i][0]} for i in range(len(dataset))]
        input_features = [{"input_features": input_case[0]} for input_case in features]
        # FOR MANUAL
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt").to(0)
        # FOR Huggingface
        # batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        # label_features = [{"input_ids": dataset_list[i][1]} for i in range(len(dataset))]
        label_features = [{"input_ids": self.processor.tokenizer(input_case[1]).input_ids} for input_case in features]
        # pad the labels to max length
        # FOR MANUAL
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt").to(0)
        # FOR Huggingface
        # labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's appended later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # TODO Try One-Hot Encoding
        batch["labels"] = labels

        # # get the tokenized label sequences
        # # label_features = [{"input_ids": dataset_list[i][1]} for i in range(len(dataset))]
        prompt_features = [{"input_ids": self.processor.tokenizer(input_case[2]).input_ids} for input_case in features]
        # # pad the labels to max length
        # FOR MANUAL
        prompts_batch = self.processor.tokenizer.pad(prompt_features, return_tensors="pt").to(0)
        # For Huggingface
        # prompts_batch = self.processor.tokenizer.pad(prompt_features, return_tensors="pt")

        # # replace padding with -100 to ignore loss correctly
        prompts = prompts_batch["input_ids"].masked_fill(prompts_batch.attention_mask.ne(1), -100)
        #
        # # if bos token is appended in previous tokenization step,
        # # cut bos token here as it's appended later anyways
        if (prompts[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            prompts = prompts[:, 1:]
        #
        if prompts.size()[1] > 448:  # Whisper Token Max:
            prompts = prompts[:, -448:]
        batch["prompts"] = prompts  # dec_input_ids, prompts

        return batch
