import os
import torch
import pandas as pd 
from typing import Any, TypeVar
from typing import List, Dict, Union
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments 
from transformers import Trainer, AutoModelForCausalLM, IntervalStrategy
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

MODEL_NAME: str = 'xhyi/PT_GPTNEO350_ATG'
BOS_TOKEN: str = '<|startoftext|>'
EOS_TOKEN: str = '<|endoftext|>'
PAD_TOKEN: str = '<|pad|>'

model_path = "./model"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = PAD_TOKEN

model = GPTNeoForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

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