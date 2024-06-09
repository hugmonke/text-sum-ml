import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler
from sklearn.model_selection import train_test_split
#import torch.nn as nn
#from torch.utils.data import DataLoader
#from torch.optim.optimizer import Optimizer
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


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.eta_min = eta_min
        self.T_max = T_max
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)
       

    def get_lr(self):
        """CosineAnnealingLR formula
        
        eta_min, eta_max - ranges for the learning rate, with eta_max being set to the inital Learning Rate
        last_epoch (T_cur) - represents the number of epochs that were run since the last restart
        T_max - the maximum number of iterations
        
        """
        return [self.eta_min + 0.5 * (eta_max - self.eta_min) * (1 + np.cos(np.pi*(self.last_epoch/self.T_max))) for eta_max in self.base_lrs]
    
    def _reset(self, epoch, T_max):
        """
        Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        return CosineAnnealingLR(self.optimizer, self.T_max, self.eta_min, last_epoch=epoch)
    
def get_seq2seq_loss(input, target):
    seq_len, batch_size = target.size()
    seq_len_input, batch_size_input, nc = input.size()
    if seq_len > seq_len_input: 
        input = F.pad(input, (0, 0, 0, 0, 0, seq_len - seq_len_input))
    input = input[:seq_len]
    return F.cross_entropy(input.reshape(-1, nc), target.reshape(-1))

def get_trainable_params(m):
    return [p for p in m.parameters() if p.requires_grad]

def format_tensor(X):
    return torch.stack([torch.tensor(x, dtype=torch.long) for x in X]).transpose(0, 1).cpu()

def validate(model, val_loader):
    n_batches = int(len(val_loader.dataset)/val_loader.batch_size)
    total_loss = 0.0
    model.eval()
    
    for _, (text, summary) in enumerate(val_loader):
        text = format_tensor(text)
        summary = format_tensor(summary)
        model_output = model(text)
        s2s_loss = get_seq2seq_loss(model_output, summary)
        total_loss += s2s_loss.item()
        
    return total_loss/n_batches    
    
def train_loop(model, 
               epoch,
               all_lr,
               optimizer,
               scheduler, 
               tr_ratios, 
               grad_clip, 
               total_loss,
               train_loader, 
               print_period=1000, 
               num_print_periods=0,
               ):
    
    n_batches = int(len(train_loader.dataset)//train_loader.batch_size)
    epoch_loss = 0.0
    model.train()
    
    for i, (text, summary) in enumerate(train_loader):
        optimizer.zero_grad()
        text = format_tensor(text)
        summary = format_tensor(summary)
        model_output = model(text, summary, tr_ratios[epoch])
        s2s_loss = get_seq2seq_loss(model_output, summary)
        epoch_loss += s2s_loss.item()
        s2s_loss.backward()
        optimizer.step()
        scheduler.step()
        all_lr.append(scheduler.get_lr())
        
        # Total norm of the parameter gradients
        clip_grad_norm_(get_trainable_params(model), grad_clip)
        
        # Prints 
        if i%print_period == 0 and i != 0:
            epoch_loss = epoch_loss / print_period
            statement = 'Epoch loss: {:.5f}. Epoch progress: {:.0f}%'.format(epoch_loss, 100*(i/n_batches))
            sys.stdout.write('\r' + statement)
            sys.stdout.flush()
            total_loss += epoch_loss
            epoch_loss = 0.0   
            num_print_periods += 1
            
    train_loss = total_loss/num_print_periods
    return train_loss
    
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def fit(model=None, 
        train_loader=None, 
        save_path=None,
        return_history=False,
        learning_rate=1e-5,
        print_period=100, 
        val_loader=None, 
        tr_ratios=None,
        grad_clip=0.0, 
        metrics=None, 
        cycle_len=1, 
        SAVE=True,  
        n_epochs=1, 
        pad=1):
    
    assert(model is not None), 'model variable cannot be None!'
    assert(train_loader is not None), 'train_loader variable cannot be None!'
    assert(save_path is not None), 'save_path variable cannot be None!'
    assert(val_loader is not None), 'val_loader variable cannot be None!'
    
    if tr_ratios is None:
        tr_ratios = np.linspace(1., 0., num=n_epochs)
    assert(n_epochs==len(tr_ratios)), 'Wrong length of tr_ratios!'
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    batches_num = int(len(train_loader.dataset)//train_loader.batch_size)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=batches_num * cycle_len)
    
    all_lr = []
    total_loss = 0.0
    best_val_loss = np.inf
    all_train_loss = []
    num_print_periods = 0
    all_validation_loss = []
 
    try: #to restore an existing checkpoint
        checkpoint = torch.load(save_path)
        print("Found an existing checkpoint!")
        print('Restoring model...')
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        print('Model restored!')
        
    except:
        pass
    
    for epoch in trange(1, n_epochs+1, desc="EPOCH: "):
        
        train_loss = train_loop(model, 
                                epoch,
                                all_lr,
                                optimizer,
                                lr_scheduler, 
                                tr_ratios, 
                                grad_clip, 
                                total_loss,
                                train_loader, 
                                print_period, 
                                num_print_periods,
                                )
        
        all_train_loss.append(train_loss)
        print_output = [epoch, train_loss]
    
        val_loss = validate(model, val_loader)
        all_validation_loss.append(val_loss)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if SAVE:
                ensure_dir(save_path)
                state = {
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }
                torch.save(state, save_path)
        print_output.append(val_loss)
        print(print_output)
        
        if epoch%cycle_len == 0:
                lr_scheduler = lr_scheduler._reset(epoch, T_max=batches_num * cycle_len)
        
        epoch += 1
        total_loss = 0.0
        num_print_periods = 0
        
    if return_history:
        history = {
                'all_lr': all_lr,
                'train_loss': all_train_loss,
                'val_loss': all_validation_loss
                }
        return history
    
     
def main(subset = False, 
        num_hidden = 256, 
        num_layers = 2, 
        activation = F.tanh, 
        em_sz_enc=100,
        max_vocab_text = 10000,
        maxlen_text = 100,
        maxlen_summary = 6,
        max_unk_text = 1,
        max_unk_summary = 1,
        DATA_FILE = "Reviews.csv",
        OUTPUT_PATH = "ProcessedData/",
        GLOVE_PATH = "glove.6B.100d.txt",
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
    test_text = all_text_model.transform(test_df.Text.values, word2idx=all_text_model.word2idx, padding='pre')

    train_summary = all_text_model.transform(train_df.Summary.values, word2idx=all_text_model.word2idx, maxlen=maxlen_summary, padding='post')
    test_summary = all_text_model.transform(test_df.Summary.values, word2idx=all_text_model.word2idx, maxlen=maxlen_summary, padding='post')
    #train_text, train_summary = limit_unk_vocab(train_text, train_summary, all_text_model, max_unk_text, max_unk_summary)
    print('Text size: ', len(train_text))
    print('Summary size: ', len(train_summary))
    print("Creating pickle files...")
    try:
        open(GLOVE_PATH, 'r', encoding='utf-8')
    except IOError:
        raise 'Missing glove file or path to file is incorrect'
    
    if not os.path.exists("glove_vectors.pkl"):
        glove_vectors = Glove.load_glove_vectors(GLOVE_PATH)
        Glove.save_glove_vectors(glove_vectors, VEC_FILE)
    with open(VEC_FILE, 'rb') as glove_pkl:
        global_vectors = pickle.load(glove_pkl)
        
    if subset > 0:
        train_loader = TextLoad.to_dataloader_(train_text[:subset], train_summary[:subset], batch_size = 32)
        test_loader = TextLoad.to_dataloader_(test_text[:subset], test_summary[:subset], batch_size = 32) 
    else:
        train_loader = TextLoad.to_dataloader_(train_text, train_summary, batch_size = 32)
        test_loader = TextLoad.to_dataloader_(test_text, test_summary, batch_size = 32)
        
    print('Training data size: ', len(train_loader)) 
    print('Testing data size: ', len(test_loader)) 
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
        
    tr_ratios = np.concatenate([np.array([0.9]*50), np.linspace(0.9, 0, 50)])
    
    # None if return_history = False
    history = fit(
                model, 
                train_loader, 
                save_path=os.path.join(OUTPUT_PATH, 'checkpoint.pth.tar'),
                return_history=False,
                learning_rate=1e-5, 
                print_period=100,
                val_loader=test_loader, 
                tr_ratios=tr_ratios,
                grad_clip=5.0, 
                SAVE=True, 
                n_epochs=len(tr_ratios), 
                pad=all_text_model.word2idx['_pad_']
                )
    print("Done!\n")    
    
if __name__ == '__main__':
    print("Launching...\n")
    subset = 40000 #set subset to 0 or less if no subset is needed
    num_hidden = 256
    num_layers = 2
    activation = F.tanh
    em_sz_enc = 100
    max_vocab_text = 10000
    maxlen_text = 100
    maxlen_summary = 5
    max_unk_text = 1
    max_unk_summary = 1
    DATA_FILE = "Reviews.csv"
    OUTPUT_PATH = "ProcessedData/"
    GLOVE_PATH = "glove.6B.100d.txt"
    VEC_FILE = "glove_vectors.pkl"
    main(subset, 
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
        OUTPUT_PATH,
        GLOVE_PATH,
        VEC_FILE)
