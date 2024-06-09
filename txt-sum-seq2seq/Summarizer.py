import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

import os
import sys
import pickle
import numpy as np
import pandas as pd

import Glove
from tqdm import trange
from Dataset import TextLoad
from Model import Seq2SeqWithAttention
from TextPreprocess import TextDataSet, limit_unk_vocab

""" Input your text in main() function """


def get_max(probs, top=3, unk=0):
        preds = []
        for top_preds in probs.topk(top)[1].data.cpu().numpy():
            top_preds = top_preds.flatten()
            final_pred = top_preds[0]
            if top > 1: 
                if top_preds[0] == unk:
                    final_pred = top_preds[1]
            preds.append(final_pred)
        return preds

def get_word2idx(data,  
                 word2idx, 
                 pad=1):
    return [word2idx[i] for i in data if i != pad]

def generate_summary(model, 
                     all_text_model,
                     text=None, 
                     pad=1, 
                     unk=0):
    
    my_example = get_word2idx(text, all_text_model.word2idx, pad=pad)
    x = np.expand_dims(my_example, 1)
    x_v = torch.from_numpy(x).long().cpu()
    probabilities = model(x_v)
    predictions = [all_text_model.idx2word[i] for i in get_max(probabilities, unk=unk) if i != pad]
    print('ORIGINAL TEXT:')
    print(text)
    print('MODEL SUMMARY:')
    print(' '.join(predictions))
    
def main(text = "WARNING: No text given!", 
        num_hidden = 256, 
        num_layers = 2, 
        activation = F.tanh, 
        em_sz_enc=100,
        max_vocab_text = 10000,
        maxlen_text = 100,
        maxlen_summary = 6,
        max_unk_text = 1,
        max_unk_summary = 0,
        DATA_FILE = "Reviews.csv",
        VEC_FILE = "glove_vectors.pkl"):
    
    reviews = pd.read_csv(DATA_FILE, usecols=['Summary', 'Text'])
    # drop rows with no Summary
    reviews.dropna(inplace=True)
    # drop duplicated 
    reviews.drop_duplicates(['Text'], keep='first', inplace=True)
    reviews['Summary_len'] = reviews.Summary.apply(lambda x: len(x.split()))
    reviews['Text_len'] = reviews.Text.apply(lambda x: len(x.split()))      
    reviews = reviews[reviews['Text_len'] > 10]
    train_df, test_df = train_test_split(reviews, test_size=0.1, random_state=100)
    # all text model
    all_text_raw = np.concatenate([train_df.Text.values, train_df.Summary.values])
    all_text_model = TextDataSet(max_vocab=max_vocab_text, maxlen=maxlen_text, min_freq=2, padding='post')
    all_text = all_text_model.fit(all_text_raw, tokenize=True)
    # Text Field
    train_text = all_text_model.transform(train_df.Text.values, word2idx=all_text_model.word2idx, padding='pre')
    
    train_summary = all_text_model.transform(train_df.Summary.values, word2idx=all_text_model.word2idx, maxlen=maxlen_summary, padding='post')
    train_text, train_summary = limit_unk_vocab(train_text, train_summary, all_text_model, max_unk_text, max_unk_summary)
    
    print("Creating pickle files...")
    try:
        with open(VEC_FILE, 'rb') as glove_pkl:
            global_vectors = pickle.load(glove_pkl)
    except IOError:
        raise 'Missing glove file or path to file is incorrect'
        
        
    model = Seq2SeqWithAttention(
                                num_hid_layers=num_hidden, 
                                vect_encoder=global_vectors, 
                                vect_decoder=global_vectors, 
                                out_seq_len=maxlen_summary,
                                num_layers=num_layers,
                                idx2word=all_text_model.idx2word, 
                                emb_size=em_sz_enc, 
                                pad_idx=all_text_model.word2idx['_pad_'],
                                activation=activation
                                ).cpu()
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    checkpoint = torch.load('ProcessedData/checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    generate_summary(model, text=text,
                     all_text_model=all_text_model,
                     pad=all_text_model.word2idx['_pad_'], 
                     unk=all_text_model.word2idx['_unk_'])       
    
if __name__ == '__main__':
    print("Launching...\n")
    
    # HERE INPUT YOUR TEXT TO SUMMARISE IT
    ##########
    
    custom_text = "I am very happy with the quality of my new hairdryer, but the price is way too big!"
    
    ##########
    num_hidden = 256
    num_layers = 2
    activation = F.tanh
    em_sz_enc = 100
    max_vocab_text = 10000
    maxlen_text = 100
    maxlen_summary = 6
    max_unk_text = 1
    max_unk_summary = 0
    DATA_FILE = "Reviews.csv"
    VEC_FILE = "glove_vectors.pkl"
    main(custom_text, 
        num_hidden,
        num_layers,
        activation,
        em_sz_enc,
        max_vocab_text,
        maxlen_text,
        maxlen_summary,
        max_unk_text,
        max_unk_summary,
        DATA_FILE,
        VEC_FILE)
