# %%
# !pip install -q -r requirements.txt

# %%
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", context="talk")
plt.style.use('dark_background')

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

import tokenizers
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, auc

# %%
INPUT_COL = 'Processed comment_text'
TARGET_COL = ['toxic', 'severe_toxic','obscene', 'threat', 'insult','identity_hate']

# %%
train_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comments-preprocessed/train.csv', nrows=100)
test_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comments-preprocessed/test.csv', nrows=100)

# %%
len(test_data)

# %%
train_data = train_data.dropna(subset=[INPUT_COL])
train_data.head()

# %%
test_data = test_data.dropna(subset=[INPUT_COL])
test_data.head()

# %%
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

# %%
# sentence_len = []
# for sentence in tqdm(train_data[INPUT_COL]):
#     token_words = tokenizer.encode_plus(sentence)['input_ids']
#     sentence_len.append(len(token_words))
    
# sns.displot(sentence_len, kde=True)
# plt.xlim([0, 300])
# plt.xlabel('Token count')

# %% [markdown]
# - Density measures the proportion of unique tokens in a text corpus.
# - Number of tokens refers to the total number of words or subword units in a text corpus.
# - Density can lead to more interpretable models, but may result in loss of information.
# - Number of tokens can preserve more information, but may lead to less interpretable models.
# - The choice between density and number of tokens depends on the specific task and the desired trade-off between model performance and interpretability.

# %%
max_len = 200
class BertDataSet(Dataset):

    def __init__(self, sentences, toxic_labels):
        self.sentences = sentences.to_numpy()
        self.targets = toxic_labels.to_numpy()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        bert_sentence = tokenizer.encode_plus(
            self.sentences[idx],
            add_special_tokens = True,
            max_length = max_len,
            pad_to_max_length = True,
            truncation = True,
            return_attention_mask = True
        )

        return {
            'ids' : torch.tensor(bert_sentence['input_ids'], dtype = torch.long),
            'mask' : torch.tensor(bert_sentence['attention_mask'], dtype = torch.long),
            'toxic_label': torch.tensor(self.targets[idx], dtype = torch.float)
        }


# %%
train_dataset = BertDataSet(train_data[INPUT_COL], train_data[TARGET_COL])
valid_dataset = BertDataSet(test_data[INPUT_COL], test_data[TARGET_COL])

# %%
train_batch = 32
valid_batch = 32

# %%
train_dataloader = DataLoader(train_dataset, batch_size = train_batch, pin_memory = True, num_workers = 2, shuffle = True)
valid_dataloader = DataLoader(valid_dataset, batch_size = valid_batch, pin_memory = True, num_workers = 2, shuffle = False)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# %%
model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 6)
model.to(device)
model.train()

# %%
for a in train_dataloader:
    ids = a['ids'].to(device)
    mask = a['mask'].to(device)
    output = model(ids, mask)
    break

# %%
output_probs = func.softmax(output['logits'], dim = 1)
torch.max(output_probs, dim = 1)

# %%
epochs = 5
LR = 2e-5 #Learning rate
optimizer = AdamW(model.parameters(), LR, betas = (0.9, 0.999), weight_decay = 1e-2, correct_bias = False)

# %%
train_steps = int((len(train_data) * epochs)/train_batch)
num_steps = int(train_steps * 0.1)
scheduler = get_linear_schedule_with_warmup(optimizer, num_steps, train_steps)

# %%
le = []
for b in tqdm(range(epochs)):
    for a in train_dataloader:
        le.append(scheduler.get_last_lr())
        scheduler.step()
plt.plot(np.arange(len(le)), le)

# %%
loss_fn = nn.BCEWithLogitsLoss()
loss_fn.to(device)
scaler = torch.cuda.amp.GradScaler()

# %%
def training(train_dataloader, model, optimizer, scheduler):
    model.train()
    torch.backends.cudnn.benchmark = True
    correct_predictions = 0

    for a in train_dataloader:
        losses = []
        optimizer.zero_grad()

        #allpreds = []
        #alltargets = []

        with torch.cuda.amp.autocast():

            ids = a['ids'].to(device, non_blocking = True)
            mask = a['mask'].to(device, non_blocking = True)

            output = model(ids, mask) #This gives model as output, however we want the values at the output
            output = output['logits'].squeeze(-1).to(torch.float32)

            output_probs = torch.sigmoid(output)
            preds = torch.where(output_probs > 0.5, 1, 0)

            toxic_label = a['toxic_label'].to(device, non_blocking = True)
            loss = loss_fn(output, toxic_label)

            losses.append(loss.item())
            #allpreds.append(output.detach().cpu().numpy())
            #alltargets.append(toxic.detach().squeeze(-1).cpu().numpy())
            correct_predictions += torch.sum(preds == toxic_label)

        scaler.scale(loss).backward() #Multiplies (‘scales’) a tensor or list of tensors by the scale factor.
                                      #Returns scaled outputs. If this instance of GradScaler is not enabled, outputs are returned unmodified.
        scaler.step(optimizer) #Returns the return value of optimizer.step(*args, **kwargs).
        scaler.update() #Updates the scale factor.If any optimizer steps were skipped the scale is multiplied by backoff_factor to reduce it.
                        #If growth_interval unskipped iterations occurred consecutively, the scale is multiplied by growth_factor to increase it
        scheduler.step() # Update learning rate schedule

    losses = np.mean(losses)
    corr_preds = correct_predictions.detach().cpu().numpy()
    accuracy = corr_preds/(len(train_data)*6)

    return losses, accuracy


# %%
def validating(valid_dataloader, model):

    model.eval()
    correct_predictions = 0
    all_output_probs = []

    for a in valid_dataloader:
        losses = []
        ids = a['ids'].to(device, non_blocking = True)
        mask = a['mask'].to(device, non_blocking = True)
        output = model(ids, mask)
        output = output['logits'].squeeze(-1).to(torch.float32)
        output_probs = torch.sigmoid(output)
        preds = torch.where(output_probs > 0.5, 1, 0)

        toxic_label = a['toxic_label'].to(device, non_blocking = True)
        loss = loss_fn(output, toxic_label)
        losses.append(loss.item())
        all_output_probs.extend(output_probs.detach().cpu().numpy())

        correct_predictions += torch.sum(preds == toxic_label)
        corr_preds = correct_predictions.detach().cpu().numpy()

    losses = np.mean(losses)
    corr_preds = correct_predictions.detach().cpu().numpy()
    accuracy = corr_preds/(len(test_data)*6)

    return losses, accuracy, all_output_probs


# %%
%%time

best_score = 1000
train_accs = []
valid_accs = []
train_losses = []
valid_losses = []

for eboch in tqdm(range(epochs)):

    train_loss, train_acc = training(train_dataloader, model, optimizer, scheduler)
    valid_loss, valid_acc, valid_probs = validating(valid_dataloader, model)

    print('train losses: %.4f' % train_loss, 'train accuracy: %.3f' % train_acc)
    print('valid losses: %.4f' % valid_loss, 'valid accuracy: %.3f' % valid_acc)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)


    if valid_loss < best_score:
        best_score = valid_loss
        print('Found a good model!')
        state = {
            'state_dict': model.state_dict(),
            'optimizer_dict': optimizer.state_dict(),
            'best_score': best_score
        }
        torch.save(state, 'best_model.pth')
    else:
        pass


