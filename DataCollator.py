# Based on https://huggingface.co/blog/fine-tune-whisper
import torch

from dataclasses import dataclass
from typing import Any, Dict

from EmergencyCallsDataset import EmergencyCallsDataset


@dataclass
class DataCollatorEmergencyCalls:
    processor: Any

    def __call__(self, features) -> Dict[str, torch.tensor]:
        # dataset = EmergencyCallsDataset()
        # dataset_list = [dataset[i] for i in range(len(dataset))]
        # input_features = [{"input_features": dataset_list[i][0]} for i in range(len(dataset))]
        input_features = [{"input_features": input_case[0]} for input_case in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        # label_features = [{"input_ids": dataset_list[i][1]} for i in range(len(dataset))]
        label_features = [{"input_ids": self.processor.tokenizer(input_case[1]).input_ids} for input_case in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's appended later anyways
        if(labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
