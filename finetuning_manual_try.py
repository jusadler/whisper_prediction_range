# Based on https://huggingface.co/blog/fine-tune-whisper and https://huggingface.co/learn/nlp-course/chapter3/4?fw=pt
# Based on https://www.youtube.com/watch?v=jF43_wj_DCQ
import datetime
import os
import random
from typing import List, Tuple

import jiwer
import numpy as np
import pandas as pd
# import evaluate
import torch
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torcheval.metrics import WordErrorRate

import whisper
from EmergencyCallsDataset import EmergencyCallsDataset
from EmergencyCallsValidationDataset import EmergencyCallsValidationDataset

# Seeds
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

# Random Split for splitting dataset? https://www.youtube.com/watch?v=jF43_wj_DCQ
# for decoder:
decoding_options = {'language': 'de'}  # => Prompt can be added here as well!!!! TODO

os.chdir("E:/Modelle/training_test/grid_search/")

metric = WordErrorRate()

model_size_ = "tiny"

number_of_layers = {
    "tiny": 4,
    "base": 6,
    "small": 12,
    "medium": 24,
    "large": 32
}

# Grid
learning_rates = [0.000001, 0.0000001, 0.00000005]
optimizers = ["Adam", "AdamW", "SGD"]
active_layers_conditions = [[f'decoder.blocks.{number_of_layers.get(model_size_) - 1}.mlp'], ['decoder.ln'],
                            [f'decoder.blocks.{number_of_layers.get(model_size_) - 1}.mlp', 'decoder.ln']]
epochs_list = [3]

eval_full_dataloader = DataLoader(
    EmergencyCallsValidationDataset("E:/Notrufe/metadata_validation.csv", path_only=True),
    batch_size=1)


def calculate_wer(model):
    vpreds = []
    ground_truth = []
    for vdata_full in eval_full_dataloader:
        vinputs_full, vlabels_full = vdata_full
        vpreds.append(model.transcribe(vinputs_full[0]).get("text"))
        ground_truth.append(vlabels_full[0])

    return jiwer.wer(ground_truth, vpreds)


def print_net_parameters(net):
    for name, para in net.named_parameters():
        print("-" * 20)
        print(f"name: {name}")
        print("values: ")
        print(para.requires_grad)


def train_one_epoch(epoch_index, tb_writer, optimizer, train_dataloader, model, loss_fn) -> Tuple[float, List]:
    running_loss = 0.
    last_loss = 0.

    performance_list = []

    for count, data in enumerate(train_dataloader):
        # inputs = data.get("input_features")
        # labels = data.get('labels')
        # prompts = data.get('prompts')  # TODO Add prompts
        inputs = data[0]
        labels = data[1]
        prompts = data[1]
        optimizer.zero_grad()
        outputs = model(inputs, prompts)
        # loss = loss_fn(torch.argmax(outputs, dim=2), labels)
        loss = loss_fn(outputs.view(-1, model.dims.n_vocab), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if count % 200 == 199:
            last_loss = running_loss / 200
            print('  batch {} loss: {}'.format(count + 1, last_loss))
            # tb_x = epoch_index * len(train_dataloader) + count + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0
            # wer = calculate_wer(model)
            performance_list.append({
                "epoch": epoch_index,
                "step_in_epoch": count + 1,
                # "wer": wer,
                "train_loss": last_loss
            })
    return last_loss, performance_list


def train_model(model_size, layer_conditions, learning_rate: float = 0.00001, epochs: int = 5,
                optimizer_name: str = "AdamW"):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    performance_list = []

    # # Load Tokenizer
    # tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language="de", task="transcribe")

    train_dataloader = DataLoader(EmergencyCallsDataset(), shuffle=True, batch_size=1)  # , collate_fn=data_collator)
    # pin_memory=True)
    eval_dataloader = DataLoader(EmergencyCallsValidationDataset("E:/Notrufe/metadata_split_validation.csv"),
                                 batch_size=1)

    model = whisper.load_model(model_size)

    loss_fn = CrossEntropyLoss()

    model.to(torch.cuda.current_device())

    # print_net_parameters(model)
    for name, param in model.named_parameters():
        # if param.requires_grad and not f'decoder.blocks.{number_of_layers.get(model_size) - 1}.mlp' in name:  # Try only using MLP of 11
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

    print("Word Error Rate on Validation Data:")
    best_wer = calculate_wer(model)
    print(best_wer)
    performance_list.append({
        "epoch": 0,
        "step_in_epoch": 0,
        "wer": best_wer
    })

    for epoch in range(epochs):
        print('EPOCH {}'.format(epoch + 1))

        model.train(True)
        avg_loss, performance_list_epoch = train_one_epoch(epoch, writer, optimizer, train_dataloader, model, loss_fn)
        model.train(False)
        performance_list += performance_list_epoch

        running_vloss = 0.0
        for i, vdata in enumerate(eval_dataloader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs, vlabels)
            # vloss = loss_fn(voutputs, vlabels)
            vloss = loss_fn(voutputs.view(-1, model.dims.n_vocab), vlabels.reshape(-1))
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        print("Word Error Rate on Validation Data:")
        avg_wer = calculate_wer(model)
        print(avg_wer)
        print(avg_wer < best_wer)
        performance_list.append({
            "epoch": epoch,
            "wer": avg_wer,
            "train_loss": avg_loss,
            "validation_loss": avg_vloss
        })

        if avg_wer < best_wer or epoch == epochs - 1:
            best_wer = avg_wer
            checkpoint_path = 'model_checkpoint_{}_{}.pt'.format(timestamp, epoch)
            torch.save({
                'epoch': epoch,
                'model_size': model_size,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
                'wer': avg_wer,
                'dims': model.dims
            }, checkpoint_path)

    performance_path = f'model_performance_{timestamp}_{model_size}_{learning_rate}_{optimizer_name}_{epochs}.csv'
    pd.DataFrame(performance_list, columns=["epoch", "wer", "train_loss", "validation_loss", "step_in_epoch"]).to_csv(
        performance_path)


for epoch_number in epochs_list:
    for lr in learning_rates:
        for active_layer_condition in active_layers_conditions:
            for optimizer_type in optimizers:
                print(f"Number of Epochs: {epoch_number}")
                print(f"Learning Rate: {lr}")
                print(f"Layer Conditions: {active_layer_condition}")
                print(f"Optimizer: {optimizer_type}")
                train_model(model_size_, active_layer_condition, learning_rate=lr, epochs=epoch_number,
                            optimizer_name=optimizer_type)
