import torch
#import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
#from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import sys
import os
import pickle
import pandas as pd
from dataset import create_dataloader
from preprocessing import TextDataSet
from model import Seq2SeqWithAttention
from tqdm import trange
import glove


"""
RUN THIS FILE TO START TRAINING THE BOT
"""

def seq2seq_loss(input, target):
    sl, bs = target.size()
    sl_in, bs_in, nc = input.size()
    if sl > sl_in: 
        input = F.pad(input, (0, 0, 0, 0, 0, sl - sl_in))
    input = input[:sl]
    return F.cross_entropy(input.reshape(-1, nc), target.reshape(-1))

def trainable_params_(m):
    return [p for p in m.parameters() if p.requires_grad]

def format_tensor(X):
    return torch.stack([torch.tensor(x, dtype=torch.long) for x in X]).transpose(0, 1).cuda()

def fit(model, train_loader, opt_fn=None, learning_rate=1e-5, 
        epochs=1, cycle_len=1, val_loader=None, metrics=None, 
        SAVE=False, save_path='ProcessedData/checkpoint.pth.tar', 
        pre_saved=False, print_period=100, grad_clip=0.0, 
        tr_ratios=None, pad=1, return_history=False):
    
    if tr_ratios is None:
        tr_ratios = np.linspace(1.0, 0.0, num=epochs)
    assert(len(tr_ratios) == epochs), 'Length of "tr_ratios" must be equal to number of epochs'
    
    if opt_fn:
        optimizer = opt_fn(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    else:  
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    n_batches = int(len(train_loader.dataset) // train_loader.batch_size)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_batches * cycle_len)
    
    all_lr = []
    all_train_loss = []
    all_valid_loss = []
    
    best_val_loss = np.inf
    
    if pre_saved:
        checkpoint = torch.load(save_path)
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Restoring model...')
        
    begin = True
    total_loss = 0.0
    num_print_periods = 0
    for epoch_ in trange(1, epochs + 1, desc='Epoch'):
        
        if pre_saved:      
            if begin:
                epoch = epoch_
                begin = False
        else:
            epoch = epoch_
        
        train_loss = train_loop(model, train_loader, optimizer, scheduler, tr_ratios, epoch_ - 1, grad_clip, 
                                print_period, total_loss, num_print_periods, all_lr)
        all_train_loss.append(train_loss)
        print_output = [epoch, train_loss]

        if val_loader:
            val_loss = validate(model, val_loader)
            all_valid_loss.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if SAVE:
                    if save_path:
                        ensure_dir(save_path)
                        state = {
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'best_val_loss': best_val_loss,
                            'optimizer': optimizer.state_dict()
                                }
                        save_checkpoint(state, save_path=save_path)
            
            print_output.append(val_loss)

        print('\n', print_output)

        if epoch_ % cycle_len == 0:
            scheduler = scheduler._reset(epoch, T_max=n_batches * cycle_len)
        
        epoch += 1
        total_loss = 0.0
        num_print_periods = 0
        
    history = {
        'all_lr': all_lr,
        'train_loss': all_train_loss,
        'val_loss': all_valid_loss,
                }

    if return_history:
        return history

def train_loop(model, train_loader, optimizer, scheduler, 
          tr_ratios, epoch, grad_clip=0.0, print_period=1000, 
          total_loss=0, num_print_periods=0, all_lr=[]):

    epoch_loss = 0.0
    n_batches = int(len(train_loader.dataset)/train_loader.batch_size)
    model.train()
    
    for i, (text, summary) in enumerate(train_loader):
        optimizer.zero_grad()
        text = format_tensor(text)
        summary = format_tensor(summary)
        output = model(text, summary, tr_ratios[epoch])
        l = seq2seq_loss(output, summary)
        epoch_loss += l.item()
        l.backward()
        optimizer.step()
        scheduler.step()
        all_lr.append(scheduler.get_lr())
        
        # Total norm of the parameter gradients
        clip_grad_norm_(trainable_params_(model), grad_clip)
        
        if i % print_period == 0 and i != 0:
            epoch_loss = epoch_loss / print_period
            statement = 'epoch_loss: {:.5f}, % of epoch: ({:.0f}%)'.format(epoch_loss, (i / n_batches) * 100.)
            sys.stdout.write('\r' + statement)
            sys.stdout.flush()
            total_loss += epoch_loss
            epoch_loss = 0.0   
            num_print_periods += 1

    train_loss = total_loss / num_print_periods
    return train_loss

def validate(model, val_loader):
    model.eval()
    n_batches = int(len(val_loader.dataset) / val_loader.batch_size)
    total_loss = 0.0

    for i, (text, summary) in enumerate(val_loader):
        text = format_tensor(text)
        summary = format_tensor(summary)
        output = model(text)
        l = seq2seq_loss(output, summary)
        total_loss += l.item()
        
    return total_loss/n_batches    

def save_checkpoint(state, save_path='checkpoint.pth.tar'):
    torch.save(state, save_path)
    
def save_model(model, save_path='checkpoint.pth.tar'):
    torch.save(model.state_dict(), save_path)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    DATA_FILE = "Reviews.csv"
    OUTPUT_PATH = "ProcessedData/"
    VEC_FILE = "glove_vectors.pkl"
    
    data = data = pd.read_csv(DATA_FILE)
    articles = data['Text'].astype(str).tolist()
    summaries = data['Summary'].astype(str).tolist()

    data_size = len(articles)
    train_size = int(0.5*data_size)
    articles = articles[:train_size]
    summaries = summaries[:train_size]
    print(f"Data loaded! Train size is: {train_size}\n")
    
    # Creates TextDataSet object
    print("Preparing data...")
    text_data = TextDataSet(max_vocab=10000, maxlen=100)
    
    # fit_transform creates idx2word, word2idx, pad_int
    # train_text and train_summary are now matrices with padding
    train_article = text_data.fit_transform(articles)
    train_summary = text_data.fit_transform(summaries)
    print("Done!\n")
    # If OUTPUT_PATH doesn't exist - mkdir
    ensure_dir(OUTPUT_PATH)
    print("Creating pickle files...")
    if not os.path.exists("glove.6B.100d.txt"):
        with open("glove.6B.100d.txt", 'w', encoding='utf-8') as f:
            pass
    if not os.path.exists("glove_vectors.pkl"):
        glove_file_path = 'glove.6B.100d.txt'
        pickle_file_path = 'glove_vectors.pkl'
        glove_vectors = glove.load_glove_vectors(glove_file_path)
        glove.save_glove_vectors(glove_vectors, pickle_file_path)
        
    with open(os.path.join(OUTPUT_PATH, 'train_text.pkl'), 'wb') as text_pkl:
        pickle.dump(train_article, text_pkl)
    with open(os.path.join(OUTPUT_PATH, 'train_summary.pkl'), 'wb') as summary_pkl:
        pickle.dump(train_summary, summary_pkl)
    with open(VEC_FILE, 'rb') as glove_pkl:
        global_vectors = pickle.load(glove_pkl)
    print("Done!\n")    
    
    train_article = [np.array(tr, dtype=np.int64) for tr in train_article]
    train_summary = [np.array(su, dtype=np.int64) for su in train_summary]
    batch_size = 32
    train_loader = create_dataloader(train_article, train_summary, batch_size=batch_size)

    # Lists of letters used to encode and decode
    idx2word_encode = text_data.idx2word 
    idx2word_decode = text_data.idx2word
    
    emb_arg_enc = 100
    emb_arg_dec = 100
    hidden_layers_num = 256
    out_seq_length = 100
    num_layers = 2
    print("Initializing model using global vectors...")
    model = Seq2SeqWithAttention(global_vectors, idx2word_encode, emb_arg_enc, 
                                 global_vectors, idx2word_decode, emb_arg_dec,
                                 hidden_layers_num, out_seq_length, num_layers=num_layers)
    print("Model initialized!\n")
    print(f"Parameters:\n Number of expected features in input/output: {emb_arg_dec}\n Number of hidden layers: {hidden_layers_num}\n Number of recurrent layers: {num_layers}\n")
    model.cuda()
    print("Training...\n")
    lr = 1e-5
    epochs = 4
    cycle_len = 1
    SAVE = True
    print(f"Training params:\n Learning rate = {lr}\n Epochs = {epochs}\n Cycle length = {cycle_len}\n Saving = {SAVE}\n")
    fit(model, train_loader, learning_rate=lr, epochs=epochs, cycle_len=cycle_len, 
        val_loader=None, SAVE=SAVE, save_path=os.path.join(OUTPUT_PATH, 'checkpoint.pth.tar'), 
        print_period=100, grad_clip=0.0)
    
    
if __name__ == "__main__":
    print("Launching...\n")
    main()
