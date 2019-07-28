import urllib.request
import os
import zipfile
import pandas as pd
import numpy as np
import pickle

data_dir = 'data/'
model_dir = 'models/'
bert_model = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin'
bert_config = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json'
bert_vocab = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt'

stsb_dataset = '''https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5'''

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
    
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
    
def download_models():
    print("Downloading bert-base-uncased")
    urllib.request.urlretrieve(bert_model, model_dir+"pytorch_model.bin")
    print("Saved as bert-base-uncased-pytorch_model.bin")

    print("Downloading config for bert-base-uncased")
    urllib.request.urlretrieve(bert_config, model_dir+"bert_config.json")
    print("Saved config")

    print("Downloading uncased vocab")
    urllib.request.urlretrieve(bert_vocab, model_dir+'vocab.txt')
    print("Saved vocab")

    print("Downloading and extracting STS-B")
    stsb_zip = data_dir+'sts_b.zip'
    urllib.request.urlretrieve(stsb_dataset, stsb_zip)
    with zipfile.ZipFile(stsb_zip) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(stsb_zip)
    print("\tCompleted!")
 
download_models()
