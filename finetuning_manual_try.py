# Based on https://huggingface.co/blog/fine-tune-whisper and https://huggingface.co/learn/nlp-course/chapter3/4?fw=pt
# Based on https://www.youtube.com/watch?v=jF43_wj_DCQ
import datetime
import os

import jiwer
# import evaluate
import torch
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torcheval.metrics import WordErrorRate

import whisper
from EmergencyCallsDataset import EmergencyCallsDataset
from EmergencyCallsValidationDataset import EmergencyCallsValidationDataset

# Random Split for splitting dataset? https://www.youtube.com/watch?v=jF43_wj_DCQ
# for decoder:
decoding_options = {'language': 'de'}  # => Prompt can be added here as well!!!! TODO

os.chdir("E:/Modelle/training_test/Manual")

metric = WordErrorRate()

model_size = "small"


class WerLoss(torch.nn.Module):

    def __int__(self):
        super(WerLoss, self).__init__()

    def forward(self, pred_ids, label_ids):
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # wer = 100 * word_error_rate(pred_str, label_str)
        metric.reset()
        metric.update(pred_str, label_str)

        # return wer
        return metric.compute()


# model_path = "E:/Modelle/large-v2.pt"  # ("E:\Modelle\large-v2.pt" "C:\\Users\\Admin\\.cache\\whisper\\large-v2.pt
# Load Feature extractor
# feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{model_size}")
#
# # Load Tokenizer
# tokenizer = WhisperTokenizer.from_pretrained(f"openai/whisper-{model_size}", language="German", task="transcribe")
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language="de", task="transcribe")
#
# # Combine extractor and tokenizer
# processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}", language="German", task="transcribe")
#
# data_collator = DataCollatorEmergencyCalls(processor=processor)

train_dataloader = DataLoader(EmergencyCallsDataset(), shuffle=True, batch_size=1)  # , collate_fn=data_collator)
# pin_memory=True)
eval_dataloader = DataLoader(EmergencyCallsValidationDataset("E:/Notrufe/metadata_split_validation.csv"), batch_size=1)
eval_full_dataloader = DataLoader(EmergencyCallsValidationDataset("E:/Notrufe/metadata_validation.csv", path_only=True), batch_size=1)
# collate_fn=data_collator)
# pin_memory=True)  # add num workers?

# for batch in train_dataloader:
#     print({k: v.shape for k, v in batch.items()})
#     break

model = whisper.load_model(model_size)
# model = whisper.load_model(model_path)
# outputs = model(**batch)
# outputs = [model.transcribe(audio=tensor) for tensor in batch.data["input_features"]]

loss_fn = CrossEntropyLoss()  # WerLoss
# loss_fn = WerLoss()

# metric = evaluate.load("wer")
model.to(torch.cuda.current_device())


def print_net_parameters(net):
    for name, para in net.named_parameters():
        print("-" * 20)
        print(f"name: {name}")
        print("values: ")
        print(para.requires_grad)


print_net_parameters(model)
for name, param in model.named_parameters():
    if param.requires_grad and not 'decoder.blocks.11.mlp' in name:  # Try only using MLP of 11
        param.requires_grad = False
print_net_parameters(model)
non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Try Grid Search and different optimizers
optimizer = torch.optim.AdamW(non_frozen_parameters, lr=0.0000001)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

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
        if count % 100 == 99:
            last_loss = running_loss / 100
            print('  batch {} loss: {}'.format(count + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + count + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0
            # if count % 100 == 99:
            #     # state_dict_path = 'E:/Modelle/training_test/v2_test_1/model_state_dict_{}_{}_{}.pt'.format(timestamp,
            #     #                                                                                            epoch,
            #     #                                                                                            count + 1)
            #     # torch.save(model.state_dict(), state_dict_path)
            #     # model_path = 'E:/Modelle/training_test/v2_test_1/model_{}_{}_{}.pt'.format(timestamp, epoch, count + 1)
            #     # torch.save(model, model_path)
            #     checkpoint_path = 'E:/Modelle/training_test/v2_test_1/model_checkpoint_{}_{}_{}.pt'.format(timestamp,
            #                                                                                                epoch,
            #                                                                                                count + 1)
            #     torch.save({
            #         'epoch': epoch,
            #         'Steps': count+1,
            #         'model_size': model_size,
            #         'model_state_dict': model.state_dict(),
            #         # 'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': loss,
            #         'dims': model.dims
            #     }, checkpoint_path)
    return last_loss


timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

EPOCHS = 5

best_vloss = 1_000_000

for epoch in range(EPOCHS):
    print('EPOCH {}'.format(epoch + 1))

    model.train(True)
    avg_loss = train_one_epoch(epoch, writer)
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(eval_dataloader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs, vlabels)
        # vloss = loss_fn(voutputs, vlabels)
        vloss = loss_fn(voutputs.view(-1, model.dims.n_vocab), vlabels.reshape(-1))
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    vpreds = []
    ground_truth = []
    for vdata_full in eval_full_dataloader:
        vinputs_full, vlabels_full = vdata_full
        vpreds.append(model.transcribe(vinputs_full[0]))
        ground_truth.append(vlabels_full)

    print("Word Error Rate on Validation Data:")
    print(jiwer.wer(ground_truth, vpreds))

    # writer.add_scalar('Training vs. Validation Loss',
    #                   {'Training': avg_loss, 'Validation': avg_vloss},
    #                   epoch + 1)
    # writer.flush()

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        # state_dict_path = 'E:/Modelle/training_test/v2_test_1/model_state_dict_{}_{}.pt'.format(timestamp, epoch)
        # torch.save(model.state_dict(), state_dict_path)
        # model_path = 'E:/Modelle/training_test/v2_test_1/model_{}_{}.pt'.format(timestamp, epoch)
        # torch.save(model, model_path)
        checkpoint_path = 'E:/Modelle/training_test/v2_test_1/model_checkpoint_{}_{}.pt'.format(timestamp,
                                                                                                epoch)
        # torch.save({
        #     'epoch': epoch,
        #     'model_size': model_size,
        #     'model_state_dict': model.state_dict(),
        #     # 'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': avg_loss,
        #     'dims': model.dims
        # }, checkpoint_path)
