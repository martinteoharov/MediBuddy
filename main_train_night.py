# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vsHRW0zzbOo6wrgmB6Ms6m5RBVcnwV5z
"""

#!pip install transformers==4.28.0 accelerate

DATA_PATH = "./diagnose_en_dataset.feather"

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
finetune = True

MODEL_NAME: str = 'EleutherAI/gpt-neo-125m'
BOS_TOKEN: str = '<|startoftext|>'
EOS_TOKEN: str = '<|endoftext|>'
PAD_TOKEN: str = '<|pad|>'
PATIENT_TOKEN: str = '<|patient|>'
DOCTOR_TOKEN: str = '<|doctor|>'

tokenizer = GPT2Tokenizer.from_pretrained("xhyi/PT_GPTNEO350_ATG")
tokenizer.pad_token = PAD_TOKEN

model_path = "./model"

if load_checkpoint:
    print("load model from checkpoint")
    model = GPTNeoForCausalLM.from_pretrained(model_path).cuda()
    model.resize_token_embeddings(len(tokenizer))
else:
    print("load pretrained GPTNeo")
    model = GPTNeoForCausalLM.from_pretrained("xhyi/PT_GPTNEO350_ATG").cuda()
    model.resize_token_embeddings(len(tokenizer))

class PatientDiagnozeDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.txt_list = txt_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.txt_list)
    
    def __getitem__(self, idx):
        txt = self.txt_list[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length = self.max_length, padding = "max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])
        return input_ids, attn_masks


"""In[7]:"""

data = pd.read_feather(DATA_PATH)
data = BOS_TOKEN + PATIENT_TOKEN + data['Patient'].values + DOCTOR_TOKEN + data['Doctor'].values + EOS_TOKEN
print(data)

"""In[8]:"""

SEQ_LEN = 1024
SAMPLE_SIZE =  int(data.shape[0] * 0.01)
_data = [el[:SEQ_LEN]  for el in data[:SAMPLE_SIZE]]

dataset = PatientDiagnozeDataset(txt_list = _data, tokenizer = tokenizer, max_length = 1024)

TRAIN_SIZE = int(len(dataset) * 0.8)
train_dataset, val_dataset = random_split(dataset, [TRAIN_SIZE, len(dataset) - TRAIN_SIZE])

"""Create output paths"""

os.makedirs('./results', exist_ok = True)
OUTPUT_DIR: str = './results'

training_args = TrainingArguments(output_dir = OUTPUT_DIR, num_train_epochs = 2, logging_steps = 5000, 
                                  save_strategy="epoch",
                                  per_device_train_batch_size=2, per_device_eval_batch_size=2, 
                                  warmup_steps=50, weight_decay=0.01, logging_dir='./logs', 
                                  evaluation_strategy="epoch",
                                 load_best_model_at_end=True)

from torch.utils.data import DataLoader

# Define batch size
batch_size = 32

# Create data loaders for train and validation datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Custom data collator function
def collate_fn(data):
    input_ids = torch.stack([item[0] for item in data])
    attention_masks = torch.stack([item[1] for item in data])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': input_ids.clone() # Labels for causal language model are the same as input_ids
    }


def train(finetune):
    if finetune:
        print("Finetuning model...")
        _trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=collate_fn)
                                                        
        _trainer.train()
        _trainer.save_model("./model")
    else:
        print("Model already finetuned; training step skipped")

train(finetune)

model.eval()
def message(input_text: str, model=model, tokenizer=tokenizer, device='cuda'):
    prompt = (BOS_TOKEN + PATIENT_TOKEN + input_text + DOCTOR_TOKEN)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=1.9,
        max_length=100,
    )
    response_text = tokenizer.batch_decode(gen_tokens)[0]
    return response_text

msg_txt = "Hello doctor, I am 48 years old. I am experiencing weak erection and difficulty in sustaining the same. This condition was observed 10 years back. Also, there is premature ejaculation. Other physical ailments that I have are, I am suffering from hypertension and taking Amlopres-L (Amlodipine and Lisinopril) for the last 10 years, high cholesterol and triglycerides. My cholesterol level is 225 and triglyceride is 200 for the last 12 years. I used to do frequent masturbation in early age.\xa0I do have erection during morning hours many times, particularly after sound sleep or if I had long walk previous day. I am having Sildenafil 25 mg or 5 mg Cialis, which is effective enough. But, I wish to get rid of tablet support and live natural way.\xa0I consulted urologist today and he prescribed me Nano-Leo capsules and Modula 5 mg for 10 days.\xa0I wish to have your second opinion on this. Please guide."
response = message(msg_txt)
print(f"Patient: {msg_txt}")
print(f"Doctor: {response}")