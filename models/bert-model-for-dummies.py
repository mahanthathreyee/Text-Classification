# ## PART II

# %% [markdown]
# You can start to run the notebook from here until the end. I have copied all the necessary parts from part I in this notebook. The objective here is to show how the model works for 5 different folds. epochs is set to 5 and the first 2000 rows of the trianing set is used. Feel free to change these parameters and see how it affects the accuracy.

# %%
import numpy as np
import pandas as pd
import os
import random
import time

import re
import string
import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", context="talk")
plt.style.use('dark_background')

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

import tokenizers
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, auc

import warnings
warnings.simplefilter('ignore')

TRAIN_MODEL = False
train = pd.read_csv('train.csv', nrows = 500_000)
test = pd.read_csv('test.csv', nrows = 100_000)
submission = pd.read_csv('sample_submission.csv')

SEED = 34
def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
random_seed(SEED)

def clean_text(text):

    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


train['clean_text'] = train['comment_text'].apply(str).apply(lambda x: clean_text(x))
test['clean_text'] = test['comment_text'].apply(str).apply(lambda x: clean_text(x))

kfold = 5
train['kfold'] = train.index % kfold

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
max_len = 200

class BertDataSet(Dataset):
    
    def __init__(self, sentences, toxic_labels):
        self.sentences = sentences
        #target is a matrix with shape [#1 x #6(toxic, obscene, etc)]
        self.targets = toxic_labels.to_numpy()
    
    def __len__(self):
        return len(self.sentences)
    
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        bert_senten = tokenizer.encode_plus(sentence, 
                                            add_special_tokens = True, # [CLS],[SEP]
                                            max_length = max_len,
                                            pad_to_max_length = True,
                                            truncation = True,
                                            return_attention_mask = True
                                             )
        ids = torch.tensor(bert_senten['input_ids'], dtype = torch.long)
        mask = torch.tensor(bert_senten['attention_mask'], dtype = torch.long)
        toxic_label = torch.tensor(self.targets[idx], dtype = torch.float)
        
        
        return {
            'ids' : ids,
            'mask' : mask,
            'toxic_label':toxic_label
        }

epochs = 5
train_batch = 32
valid_batch = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = nn.BCEWithLogitsLoss()
loss_fn.to(device)
scaler = torch.cuda.amp.GradScaler()

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
    accuracy = corr_preds/(len(p_train)*6)
    
    return losses, accuracy

def validating(valid_dataloader, model, n):
    
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
    accuracy = corr_preds/(n*6)
    
    return losses, accuracy, all_output_probs

# %% [markdown]
# ## 8. Repeat training for k-fold

# %% [markdown]
# To improve our model we repeat the same process of training for each fold of k-folds.

# %%
best_scores = []

def train_bert_model():
    for fold in tqdm(range(0,5)):

        # initializing the data
        p_train = train[train['kfold'] != fold].reset_index(drop = True)
        p_valid = train[train['kfold'] == fold].reset_index(drop = True)

        train_dataset = BertDataSet(p_train['clean_text'], p_train[['toxic', 'severe_toxic','obscene', 'threat', 'insult','identity_hate']])
        valid_dataset = BertDataSet(p_valid['clean_text'], p_valid[['toxic', 'severe_toxic','obscene', 'threat', 'insult','identity_hate']])

        train_dataloader = DataLoader(train_dataset, batch_size = train_batch, shuffle = True, num_workers = 4, pin_memory = True)
        valid_dataloader = DataLoader(valid_dataset, batch_size = valid_batch, shuffle = False, num_workers = 4, pin_memory = True)

        model = transformers.BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels = 6)
        model.to(device)
        
        LR = 2e-5
        optimizer = AdamW(model.parameters(), LR,betas = (0.9, 0.999), weight_decay = 1e-2) # AdamW optimizer

        train_steps = int(len(p_train)/train_batch * epochs)
        num_steps = int(train_steps * 0.1)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_steps, train_steps)
        
        best_score = 1000
        train_accs = []
        valid_accs = []
        train_losses = []
        valid_losses = []
        best_valid_probs = []
        
        print("-------------- Fold = " + str(fold) + "-------------")
        
        for epoch in tqdm(range(epochs)):
            print("-------------- Epoch = " + str(epoch) + "-------------")

            train_loss, train_acc = training(train_dataloader, model, optimizer, scheduler)
            valid_loss, valid_acc, valid_probs = validating(valid_dataloader, model, len(p_valid))

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            
            print('train losses: %.4f' %(train_loss), 'train accuracy: %.3f' %(train_acc))
            print('valid losses: %.4f' %(valid_loss), 'valid accuracy: %.3f' %(valid_acc))

            if (valid_loss < best_score):

                best_score = valid_loss
                print("Found an improved model! :)")

                state = {'state_dict': model.state_dict(),
                        'optimizer_dict': optimizer.state_dict(),
                        'best_score':best_score
                        }

                torch.save(state, "model" + str(fold) + ".pth")
                best_valid_prob = valid_probs
                torch.cuda.memory_summary(device = None, abbreviated = False)
            else:
                pass


        best_scores.append(best_score)
        best_valid_probs.append(best_valid_prob)
        
        ##Plotting the result for each fold
        x = np.arange(epochs)
        fig, ax = plt.subplots(1, 2, figsize = (15,4))
        ax[0].plot(x, train_losses)
        ax[0].plot(x, valid_losses)
        ax[0].set_ylabel('Losses', weight = 'bold')
        ax[0].set_xlabel('Epochs')
        ax[0].grid(alpha = 0.3)
        ax[0].legend(labels = ['train losses', 'valid losses'])

        ax[1].plot(x, train_accs)
        ax[1].plot(x, valid_accs)
        ax[1].set_ylabel('Accuracy', weight = 'bold')
        ax[1].set_xlabel('Epochs')
        ax[1].legend(labels = ['train acc', 'valid acc'])

        ax[1].grid(alpha = 0.3)
        fig.suptitle('Fold = '+str(fold), weight = 'bold') 

    print('Mean of',kfold, 'folds for best loss in', epochs, 'epochs cross-validation folds is %.4f.' %(np.mean(best_scores)))

if TRAIN_MODEL:
    train_bert_model()
else:
    model = transformers.BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels = 6)
    model.to(device)

print('Calculating test accuracy for each best model per fold')
def predicting(test_dataloader, model, pthes):    
    for pth in pthes:
        state = torch.load(pth)
        model.load_state_dict(state['state_dict'])
        model.to(device)
        model.eval()
        with torch.no_grad():
            valid_loss, valid_acc, _valid_probs = validating(test_dataloader, model, len(test))
            print('Test losses: %.4f' %(valid_loss), 'Test accuracy: %.3f' %(valid_acc))

pthes = [os.path.join("./",s) for s in os.listdir("./") if ".pth" in s]

test_dataset = BertDataSet(test['clean_text'], test[['toxic', 'severe_toxic','obscene', 'threat', 'insult','identity_hate']])
test_dataloader = DataLoader(test_dataset, batch_size = valid_batch, shuffle = False, num_workers = 4, pin_memory = True)

predicting(test_dataloader, model, pthes)
