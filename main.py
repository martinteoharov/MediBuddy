#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
# PartialState is missing in the newest transformers
#!pip install transformers==4.28.0
#!pip install accelerate -U
#!pip install pyarrow

# In[2]:


import os
import torch
import pandas as pd 
from typing import Any, TypeVar
from typing import List, Dict, Union
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments 
from transformers import Trainer, AutoModelForCausalLM, IntervalStrategy
from transformers import GPTNeoForCausalLM, GPT2Tokenizer


load_checkpoint = False
finetune = False


# In[3]:


torch.manual_seed(2137)


# In[4]:

# Assign values to few params 
#MODEL_NAME: str = 'xhyi/PT_GPTNEO350_ATG'
MODEL_NAME: str = 'EleutherAI/gpt-neo-125m'
BOS_TOKEN: str = '<|startoftext|>'
EOS_TOKEN: str = '<|endoftext|>'
PAD_TOKEN: str = '<|pad|>'

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = PAD_TOKEN

# Load tokenizer 
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, bos_token = BOS_TOKEN, eos_token = EOS_TOKEN, pad_token = PAD_TOKEN, padding_side = "left")


# In[5]:

model_path = "./model"

if load_checkpoint:
    print("load model from checkpoint")
    model = GPTNeoForCausalLM.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
else:
    print("load pretrained GPTNeo")
    model = GPTNeoForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))


# In[6]:


class PatientDiagnozeDataset(Dataset):
    
    def __init__(self, txt_list, tokenizer, max_length):
        
        self.input_ids: List = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer(BOS_TOKEN + txt + EOS_TOKEN, truncation=True, 
                                      max_length = max_length, padding = "max_length")
            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


# In[7]:


# Load the data
#DATA_PATH = 'archive/diagnose_en_dataset.feather'
DATA_PATH = 'diagnose_en_dataset.feather'
data = pd.read_feather(DATA_PATH)
data = data['Patient'].values + EOS_TOKEN + data['Doctor'].values
print(data)


# In[8]:

SEQ_LEN = 1024
SAMPLE_SIZE =  int(data.shape[0] * 0.01)
_data = [el[:SEQ_LEN]  for el in data[:SAMPLE_SIZE]]


dataset = PatientDiagnozeDataset(txt_list = _data, tokenizer = tokenizer, max_length = 1024)

# Spit the data
TRAIN_SIZE = int(len(dataset) * 0.8)
train_dataset, val_dataset = random_split(dataset, [TRAIN_SIZE, len(dataset) - TRAIN_SIZE])


# In[9]:

# Create output paths
os.makedirs('./results', exist_ok = True)
OUTPUT_DIR: str = './results'


# In[10]:

training_args = TrainingArguments(output_dir = OUTPUT_DIR, num_train_epochs = 2, logging_steps = 5000, 
                                  save_strategy="epoch",
                                  per_device_train_batch_size=2, per_device_eval_batch_size=2, 
                                  warmup_steps=50, weight_decay=0.01, logging_dir='./logs', 
                                  evaluation_strategy="epoch",
                                 load_best_model_at_end=True)


# In[11]:

def train(finetune):
    if finetune:
        print("Finetuning model...")
        _trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                'attention_mask': torch.stack([f[1] for f in data]),
                                                                'labels': torch.stack([f[0] for f in data])})
        _trainer.train()
        _trainer.save_model("./model")
    else:
        print("Model already finetuned; training step skipped")

train(finetune)

model.eval()
def message(input_text: str, model=model, tokenizer=tokenizer, device='cpu'):
    prompt = (input_text)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )

    response_text = tokenizer.batch_decode(gen_tokens)[0]

    return response_text


msg_txt = "Give me a cure for the common flu, doctor."
response = message(msg_txt)
print(f"Patient: {msg_txt}")
print(f"Doctor: {response}")

from flask import Flask, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/process', methods=['POST'])
def process_text():
    text = request.form.get('text')
    # Process the text variable here
    # For example, you can print it
    response = message(text)
    return response

if __name__ == '__main__':
    app.run()


# In[13]:


#generated = tokenizer(BOS_TOKEN, return_tensors="pt").input_ids.cuda()


# In[18]:


#sample_outputs = model.generate(generated, do_sample=True, top_k=50,
#                               bos_token='<|startoftext|>',
#                               eos_token='<|endoftext|>', pad_token='<|pad|>',
#                               max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=20)

