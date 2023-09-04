# Based on https://www.youtube.com/watch?v=jF43_wj_DCQ,
# https://colab.research.google.com/drive/1P4ClLkPmfsaKn2tBbRp0nVjGMRKR-EWz?usp=sharing#scrollTo=m-Wq6FQQAQ3u
# link from https://github.com/openai/whisper/discussions/64,
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
import datetime
import os
import random
from typing import List, Tuple, Union, Dict

import jiwer
import numpy as np
import pandas as pd
# import evaluate
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torcheval.metrics import WordErrorRate

import whisper
from EmergencyCallsValidationDataset import EmergencyCallsValidationDataset

# Seeds
# random_seeds = [42, 16, 420, 314159]
random_seeds = [42]
torch.manual_seed(random_seeds[0])
random.seed(random_seeds[0])
np.random.seed(random_seeds[0])

# Random Split for splitting dataset? https://www.youtube.com/watch?v=jF43_wj_DCQ
# for decoder:
decoding_options = {'language': 'de'}  # => Prompt can be added here as well!!!! TODO
path = "E:/Modelle/Full Training/"
os.chdir(path)

metric = WordErrorRate()

model_sizes = ["medium", "large-v1"]  # ["large", "base", "small", "medium", "tiny"]

number_of_layers = {
    "tiny": 4,
    "base": 6,
    "small": 12,
    "medium": 24,
    "large": 32,
    "large-v1": 32
}

# Grid
learning_rates = [0.0001]  # [0.0001, 0.00001, 0.000001]  # [0.0005, 0.0001, 0.00005, 0.00001]
scheduler_values = [[1, 0.8], [1, 0.7], [1, 0.6], [1, 0.5]]  # [[2, 0.9], [1, 0.9], [1, 0.8], [1, 1]]
# TODO Try heavier decay (0.7 or lower) => Also for smaller models!!
# optimizers = ["Adam", "AdamW", "SGD"]
optimizers = ["AdamW"]
# active_layers_conditions = [[f'decoder.blocks.{number_of_layers.get(model_size_) - 1}.mlp'], ['decoder.ln'],
#                             [f'decoder.blocks.{number_of_layers.get(model_size_) - 1}.mlp', 'decoder.ln']]
batch_sizes = [16]
epochs_list = [10]


class EmergencyCallsDatasetLocal(torch.utils.data.Dataset):
    # TODO NEED TO ADD PROMPT??!
    def __init__(self, annotations_path, tokenizer, data_to_gpu=True) -> None:
        super().__init__()

        self.annotations = pd.read_csv(annotations_path, index_col=0)
        self.sample_rate = 16000
        self.tokenizer = tokenizer
        self.data_to_gpu = data_to_gpu

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, id):
        annotation = self.annotations.iloc[id]
        audio_sample_path = annotation.iloc[0]
        transcription = annotation.iloc[1]
        prompt = annotation.iloc[2]

        # audio
        mel = whisper.log_mel_spectrogram(audio_sample_path)
        if self.data_to_gpu:
            mel.to(0)

        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(transcription)
        labels = text[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text
        }

    def __getitems__(self, indices):
        return_list = []
        for index in indices:
            return_list.append(self.__getitem__(index))
        return return_list


class DataCollatorEmergencyCallsDataset:
    def __init__(self, tensors_to_gpu=True):
        super().__init__()
        self.tensors_to_gpu = tensors_to_gpu

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
        if self.tensors_to_gpu:
            input_ids = torch.concat([input_id[None, :] for input_id in input_ids]).to(0)
        else:
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

        if self.tensors_to_gpu:
            batch = {k: torch.tensor(np.array(v), requires_grad=False).type(torch.LongTensor).to(0) for k, v in
                     batch.items()}
        else:
            batch = {k: torch.tensor(np.array(v), requires_grad=False).type(torch.LongTensor) for k, v in
                     batch.items()}
        batch["input_ids"] = input_ids

        return batch


train_full_dataloader = DataLoader(
    EmergencyCallsValidationDataset("E:/Notrufe/metadata.csv", path_only=True, data_to_gpu=False)
)

eval_full_dataloader = DataLoader(
    EmergencyCallsValidationDataset("E:/Notrufe/metadata_validation.csv", path_only=True, data_to_gpu=False),
    batch_size=1)

test_full_dataloader = DataLoader(
    EmergencyCallsValidationDataset("E:/Notrufe/metadata_test.csv", path_only=True, data_to_gpu=False),
    batch_size=1)


def calculate_wer_and_cer(model, validation=True, test=True):
    preds = []
    ground_truth = []
    model.eval()
    if validation:
        for vdata_full in eval_full_dataloader:
            vinputs_full, vlabels_full = vdata_full
            vinputs_full = vinputs_full[0]
            vlabels_full = vlabels_full[0]
            preds.append(model.transcribe(vinputs_full).get("text"))
            ground_truth.append(vlabels_full)
    elif test:
        for tdata_full in test_full_dataloader:
            tinputs_full, tlabels_full = tdata_full
            tinputs_full = tinputs_full[0]
            tlabels_full = tlabels_full[0]
            preds.append(model.transcribe(tinputs_full).get("text"))
            ground_truth.append(tlabels_full)
    else:
        for tdata_full in train_full_dataloader:
            tinputs_full, tlabels_full = tdata_full
            tinputs_full = tinputs_full[0]
            tlabels_full = tlabels_full[0]
            preds.append(model.transcribe(tinputs_full).get("text"))
            ground_truth.append(tlabels_full)

    return jiwer.wer(ground_truth, preds), jiwer.cer(ground_truth, preds)


def calculate_vloss(model, eval_dataloader, loss_fn):
    running_vloss = 0.0
    with torch.no_grad():
        for i, vdata in enumerate(eval_dataloader):
            vinputs = vdata.get("input_ids").to(0)
            vlabels = vdata.get('labels').to(0)
            vdec_input_ids = vdata.get('dec_input_ids').to(0)
            voutputs = model(vinputs, vdec_input_ids)
            vloss = loss_fn(voutputs.view(-1, model.dims.n_vocab), vlabels.reshape(-1))
            running_vloss += vloss
            vinputs.detach()
            vlabels.detach()
            vdec_input_ids.detach()
            torch.cuda.empty_cache()

    return running_vloss / (i + 1)


def print_net_parameters(net):
    for name, para in net.named_parameters():
        print("-" * 20)
        print(f"name: {name}")
        print("values: ")
        print(para.requires_grad)


def train_one_epoch(epoch_index, optimizer, train_dataloader, model, loss_fn, scheduler) -> Tuple[float, List]:
    running_loss = 0.

    performance_list = []
    model.train()
    for count, data in enumerate(train_dataloader):
        inputs = data.get("input_ids")
        labels = data.get('labels')
        dec_input_ids = data.get('dec_input_ids')
        optimizer.zero_grad()
        outputs = model(inputs, dec_input_ids)
        loss = loss_fn(outputs.view(-1, model.dims.n_vocab), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # if count % test_interval == test_interval - 1:
        #     model.eval()
        #     last_loss = running_loss / test_interval
        #     print('  batch {} loss: {}'.format(count + 1, last_loss))
        #     running_loss = 0
        #     wer, cer = calculate_wer_and_cer(model)
        #     print("WER on validation data")
        #     print(wer)
        #     print("CER on validation data")
        #     print(cer)
        #     performance_list.append({
        #         "epoch": epoch_index,
        #         "step_in_epoch": count + 1,
        #         "wer": wer,
        #         "cer": cer,
        #         "train_loss": last_loss
        #     })
        #     model.train()
    scheduler.step()
    last_loss = running_loss / len(train_dataloader)
    return last_loss, performance_list


def train_model(model_size, layer_conditions, learning_rate: float = 0.00001, epochs: int = 5,
                optimizer_name: str = "AdamW", batch_size=4, scheduler_rates=None):
    if scheduler_rates is None:
        scheduler_rates = [1, 1]
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    performance_list = []

    # # Load Tokenizer
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language="de", task="transcribe")

    # train_dataloader = DataLoader(EmergencyCallsDataset(), shuffle=True, batch_size=4,
    #                               collate_fn=DataCollatorEmergencyCallsDataset())
    train_dataloader = DataLoader(EmergencyCallsDatasetLocal("E:/Notrufe/metadata_split.csv", tokenizer), shuffle=True,
                                  batch_size=batch_size, collate_fn=DataCollatorEmergencyCallsDataset())
    # pin_memory=True)
    eval_dataloader = DataLoader(EmergencyCallsDatasetLocal("E:/Notrufe/metadata_split_validation.csv", tokenizer),
                                 batch_size=1, collate_fn=DataCollatorEmergencyCallsDataset(tensors_to_gpu=False))

    model = whisper.load_model(model_size)

    loss_fn = CrossEntropyLoss(ignore_index=-100)

    model.to(torch.cuda.current_device())

    # print_net_parameters(model)
    for name, param in model.named_parameters():
        if param.requires_grad and not any(layer_condition in name for layer_condition in layer_conditions):
            param.requires_grad = False
    # print_net_parameters(model)
    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(non_frozen_parameters, lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(non_frozen_parameters, lr=learning_rate)
    else:
        optimizer = torch.optim.AdamW(non_frozen_parameters, lr=learning_rate)  # 0.0000001

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_rates[0], gamma=scheduler_rates[1])
    best_wer_epoch = -1
    print("Word Error Rate on Validation Data:")
    best_wer, best_cer = calculate_wer_and_cer(model)
    print(best_wer)
    print("CER on validation data")
    print(best_cer)
    print("Vloss:")
    vloss_start = calculate_vloss(model, eval_dataloader, loss_fn).item()
    print(vloss_start)
    print("WER on test data:")
    test_wer_start, test_cer_start = calculate_wer_and_cer(model, validation=False)
    print(test_wer_start)
    print("CER on test data:")
    print(test_cer_start)
    print("WER on train data:")
    train_wer_start, train_cer_start = calculate_wer_and_cer(model, validation=False, test=False)
    print(train_wer_start)
    print("CER on train data:")
    print(train_cer_start)
    performance_list.append({
        "epoch": 0,
        "step_in_epoch": 0,
        "val_wer": best_wer,
        "val_cer": best_cer,
        "test_wer": test_wer_start,
        "test_cer": test_cer_start,
        "train_wer": train_wer_start,
        "train_cer": train_cer_start,
        "validation_loss": vloss_start
    })

    for epoch in range(epochs):
        print('EPOCH {}'.format(epoch + 1))

        model.train(True)
        avg_loss, performance_list_epoch = train_one_epoch(epoch, optimizer, train_dataloader, model, loss_fn,
                                                           scheduler)
        model.train(False)
        model.eval()
        performance_list += performance_list_epoch

        avg_vloss = calculate_vloss(model, eval_dataloader, loss_fn).item()
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        print("Word Error Rate on Validation Data:")
        avg_wer, avg_cer = calculate_wer_and_cer(model)
        print(avg_wer)
        print("CER on validation data")
        print(avg_cer)
        print(avg_wer < best_wer)
        print("WER on train data:")
        train_wer, train_cer = calculate_wer_and_cer(model, validation=False, test=False)
        print(train_wer)
        print("CER on train data:")
        print(train_cer)
        print("WER on test data:")
        test_wer, test_cer = calculate_wer_and_cer(model, validation=False)
        print(test_wer)
        print("CER on test data:")
        print(test_cer)
        performance_list.append({
            "epoch": epoch,
            "val_wer": avg_wer,
            "val_cer": avg_cer,
            "train_wer": train_wer,
            "train_cer": train_cer,
            "test_wer": test_wer,
            "test_cer": test_cer,
            "train_loss": avg_loss,
            "validation_loss": avg_vloss
        })

        if avg_wer < best_wer or epoch == epochs - 1:
            if epoch == epochs - 1:
                last_model_best = avg_wer < best_wer or best_wer_epoch < 0
                best_wer = avg_wer
                if last_model_best:
                    test_wer, test_cer = calculate_wer_and_cer(model, validation=False)
                else:
                    best_model = whisper.load_model(f"{path}model_checkpoint_{timestamp}_{best_wer_epoch}.pt",
                                                    local_model=True)
                    test_wer, test_cer = calculate_wer_and_cer(best_model, validation=False)
                print(f"Test WER = {test_wer}; Test CER = {test_cer}")
                performance_list.append({"test_cer": test_cer, "test_wer": test_wer})
                checkpoint_path = 'model_checkpoint_{}_{}.pt'.format(timestamp, epoch)
                torch.save({
                    'epoch': epoch,
                    'model_size': model_size,
                    'model_state_dict': model.state_dict(),
                    'loss': avg_loss,
                    'val_wer': avg_wer,
                    'val_cer': avg_cer,
                    'test_wer': test_wer,
                    'test_cer': test_cer,
                    'dims': model.dims,
                    'active_settings': layer_conditions,
                }, checkpoint_path)
            else:
                best_wer = avg_wer
                checkpoint_path = 'model_checkpoint_{}_{}.pt'.format(timestamp, epoch)
                torch.save({
                    'epoch': epoch,
                    'model_size': model_size,
                    'model_state_dict': model.state_dict(),
                    'loss': avg_loss,
                    'val_wer': avg_wer,
                    'val_cer': avg_cer,
                    'dims': model.dims,
                    'active_settings': layer_conditions
                }, checkpoint_path)
                best_wer_epoch = epoch

    performance_path = f'model_performance_{timestamp}_{model_size}_{learning_rate}_{optimizer_name}_{epochs}_{batch_size}_{random_seed}_decay{int(10 * scheduler_rates[1])}.csv'
    pd.DataFrame(performance_list,
                 columns=["epoch", "step_in_epoch", "train_wer", "train_cer", "val_wer", "val_cer", "train_loss",
                          "validation_loss", "test_wer", "test_cer"]).to_csv(
        performance_path)


for model_size_ in model_sizes:
    active_layers_conditions = [[f'decoder.blocks.{number_of_layers.get(model_size_) - 1}.mlp']]  # ,
    # [f'decoder.blocks.{number_of_layers.get(model_size_) - 1}.mlp.2']]
    for random_seed in random_seeds:
        for epoch_number in epochs_list:
            for active_layer_condition in active_layers_conditions:
                for lr in learning_rates:
                    for scheduler_value in scheduler_values:
                        for optimizer_type in optimizers:
                            for train_batch_size in batch_sizes:
                                torch.manual_seed(random_seed)
                                random.seed(random_seed)
                                np.random.seed(random_seed)
                                print(f"Number of Epochs: {epoch_number}")
                                print(f"Learning Rate: {lr}")
                                print(f"Batch_size: {train_batch_size}")
                                print(f"Layer Conditions: {active_layer_condition}")
                                print(f"Optimizer: {optimizer_type}")
                                train_model(model_size_, active_layer_condition, learning_rate=lr, epochs=epoch_number,
                                            optimizer_name=optimizer_type, batch_size=train_batch_size,
                                            scheduler_rates=scheduler_value)
