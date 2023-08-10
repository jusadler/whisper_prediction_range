# Based on https://colab.research.google.com/drive/1P4ClLkPmfsaKn2tBbRp0nVjGMRKR-EWz?usp=sharing
# link from https://github.com/openai/whisper/discussions/64

# IMPORTS #
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer

import numpy as np

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import torch
import whisper
import torchaudio
import torchaudio.transforms as at
from pytorch_lightning import seed_everything
from pytorch_lightning import LightningModule
from torch import nn
import evaluate

from EmergencyCallsDataset import EmergencyCallsDataset
from EmergencyCallsValidationDataset import EmergencyCallsValidationDataset
from torch.utils.data import DataLoader
import pandas as pd

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

SAMPLE_RATE = 16000
BATCH_SIZE = 2
TRAIN_RATE = 0.8

ANNOTATION_PATH = "E:/Notrufe/metadata_split.csv"
ANNOTATION_PATH_EVAL = "E:/Notrufe/metadata_split_validation.csv"
AUDIO_MAX_LENGTH = 480000
TEXT_MAX_LENGTH = 120  ## IS THIS THE CASE FOR US??????
SEED = 3407
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
seed_everything(SEED, workers=True)


# UTIL #


def load_wave(wave_path, sample_rate: int = 16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


# DATA LOADER #

woptions = whisper.DecodingOptions(language="de", without_timestamps=True)
wmodel = whisper.load_model("large")
wtokenizer = whisper.tokenizer.get_tokenizer(True, language="de", task=woptions.task)

train_dataset = EmergencyCallsDataset()
eval_dataset = EmergencyCallsValidationDataset()


class EmergencyCallsDatasetLocal(torch.utils.data.Dataset):
    # TODO NEED TO ADD PROMPT??!
    def __init__(self, annotations_path, tokenizer) -> None:
        super().__init__()

        self.annotations = pd.read_csv(annotations_path, index_col=0)
        self.sample_rate = 16000
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, id):
        annotation = self.annotations.iloc[id]
        audio_sample_path = annotation.iloc[0]
        transcription = annotation.iloc[1]
        prompt = annotation.iloc[2]

        # audio
        audio = load_wave(audio_sample_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)

        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(transcription)
        labels = text[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text
        }


class WhisperDataCollatorWhithPadding:
    def __call__(self, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in
                  zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in
                         zip(dec_input_ids, dec_input_ids_length)]  # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids

        return batch


# CONFIRM DATA LOADING #

dataset = EmergencyCallsDatasetLocal(annotations_path=ANNOTATION_PATH, tokenizer=wtokenizer)
loader = DataLoader(dataset, batch_size=1, collate_fn=WhisperDataCollatorWhithPadding())
for b in loader:
    print(b["labels"].shape)
    print(b["input_ids"].shape)
    print(b["dec_input_ids"].shape)

    for token, dec in zip(b["labels"], b["dec_input_ids"]):
        token[token == -100] = wtokenizer.eot
        text = wtokenizer.decode(token)
        print(text)

        dec[dec == -100] = wtokenizer.eot
        text = wtokenizer.decode(dec)
        print(text)

    break

with torch.no_grad():
    audio_features = wmodel.encoder(b["input_ids"].cuda())
    input_ids = b["input_ids"]
    labels = b["labels"].long()
    dec_input_ids = b["dec_input_ids"].long()

    audio_features = wmodel.encoder(input_ids.cuda())
    print(dec_input_ids)
    print(input_ids.shape, dec_input_ids.shape, audio_features.shape)
    print(audio_features.shape)
    print()
out = wmodel.decoder(dec_input_ids.cuda(), audio_features)

print(out.shape)
print(out.view(-1, out.size(-1)).shape)
print(b["labels"].view(-1).shape)

tokens = torch.argmax(out, dim=2)
for token in tokens:
    token[token == -100] = wtokenizer.eot
    # TODO WHY DOES DECODE NOT KNOW skip_special_tokens? => Add manually??
    text = wtokenizer.decode(token)
    print(text)


# TRAINER #


class Config:
    learning_rate = 0.000005
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2
    batch_size = 16
    num_worker = 2
    num_train_epochs = 5
    gradient_accumulation_steps = 1
    sample_rate = SAMPLE_RATE


class WhisperModelModule(LightningModule):
    def __init__(self, cfg: Config, model_name="base", lang="de") -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="de", task=self.options.task)

        # only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")
        self.train_dataset = EmergencyCallsDatasetLocal(ANNOTATION_PATH, self.tokenizer)
        self.eval_dataset = EmergencyCallsDatasetLocal(ANNOTATION_PATH_EVAL, self.tokenizer)
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
            l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)

        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.cfg.learning_rate)
                          #eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1, last_epoch=-1, verbose=False)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=self.cfg.warmup_steps,
        #     num_training_steps=self.t_total
        # )
        # scheduler = None
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""

        if stage == 'fit' or stage is None:
            self.t_total = (
                    (len(self.train_dataset) // (self.cfg.batch_size))
                    // self.cfg.gradient_accumulation_steps
                    * float(self.cfg.num_train_epochs)
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.cfg.batch_size,
                                           drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                                           collate_fn=WhisperDataCollatorWhithPadding()
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.eval_dataset,
                                           batch_size=self.cfg.batch_size,
                                           num_workers=self.cfg.num_worker,
                                           collate_fn=WhisperDataCollatorWhithPadding()
                                           )


# MAIN CODE #

log_output_dir = "/content/logs"
check_output_dir = "/content/artifacts"

train_name = "whisper"
train_id = "00001"

model_name = "base"
lang = "de"

cfg = Config()

Path("E:/Modelle/training_test/v2_test_1/logs").mkdir(exist_ok=True)
Path("E:/Modelle/training_test/v2_test_1").mkdir(exist_ok=True)

tflogger = TensorBoardLogger(
    save_dir=log_output_dir,
    name=train_name,
    version=train_id
)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{check_output_dir}/checkpoint",
    filename="checkpoint-{epoch:04d}",
    save_top_k=-1 # all model save
)

callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
model = WhisperModelModule(cfg, model_name, lang, )


trainer = Trainer(
    precision=16,
    accelerator=DEVICE,
    max_epochs=cfg.num_train_epochs,
    accumulate_grad_batches=cfg.gradient_accumulation_steps,
    logger=tflogger,
    callbacks=callback_list
)

trainer.fit(model)
