import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import numpy as np
import sys
import os
import pickle

from dataset import load_data, create_dataloader
from preprocessing import TextDataSet
from model import Seq2SeqAttention
from tqdm import tqdm, trange

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.optimizer = optimizer
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2 for base_lr in self.base_lrs]
    
    def _reset(self, epoch, T_max):
        """
        Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        return CosineAnnealingLR(self.optimizer, self.T_max, self.eta_min, last_epoch=epoch)

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

def fit(model, train_loader, opt_fn=None, learning_rate=1e-5, epochs=1, cycle_len=1, val_loader=None, metrics=None, 
        save=False, save_path='models/checkpoint.pth.tar', pre_saved=False, print_period=100, grad_clip=0.0, 
        tr_ratios=None, pad=1, return_history=False):
    
    if tr_ratios is None:
        tr_ratios = np.linspace(1.0, 0.0, num=epochs)
    assert(len(tr_ratios) == epochs), 'need to have same len of "tr_ratios" as number of epochs'
    
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
        print('...restoring model...')
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
        
        train_loss = train(model, train_loader, optimizer, scheduler, tr_ratios, epoch_ - 1, grad_clip, 
                           print_period, total_loss, num_print_periods, all_lr)
        all_train_loss.append(train_loss)
        print_output = [epoch, train_loss]

        if val_loader:
            val_loss = validate(model, val_loader)
            all_valid_loss.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save:
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

def train(model, train_loader, optimizer, scheduler, tr_ratios, epoch_, grad_clip=0.0, print_period=1000, 
          total_loss=0, num_print_periods=0, all_lr=[]):

    epoch_loss = 0.
    n_batches = int(len(train_loader.dataset) / train_loader.batch_size)
    model.train()
    
    for i, (text, summary) in enumerate(train_loader):
        optimizer.zero_grad()
        text = format_tensor(text)
        summary = format_tensor(summary)
        
        output = model(text, summary, tr_ratios[epoch_])
        l = seq2seq_loss(output, summary)
        epoch_loss += l.item()
        l.backward()
        optimizer.step()
        scheduler.step()
        all_lr.append(scheduler.get_lr())
        
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
        
    return total_loss / n_batches    

def save_checkpoint(state, save_path='models/checkpoint.pth.tar'):
    torch.save(state, save_path)
    
def save_model(model, save_path='models/checkpoint.pth.tar'):
    torch.save(model.state_dict(), save_path)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    DATA_FILE = "C:/Users/blasz/OneDrive/Pulpit/summarization_proj/Reviews.csv"
    OUTPUT_PATH = "C:/Users/blasz/OneDrive/Pulpit/summarization_proj/ProcessedData"
    VEC_FILE = "glove_vectors.pkl"
    
    data = load_data(DATA_FILE)
    articles = data['Text'].astype(str).tolist()
    summaries = data['Summary'].astype(str).tolist()

    data_size = len(articles)
    train_size = int(0.8*data_size)
    articles = articles[:train_size]
    summaries = summaries[:train_size]
    
    text_data = TextDataSet(max_vocab=10000, maxlen=100)
    train_text = text_data.fit_transform(articles)
    train_summary = text_data.fit_transform(summaries)
    
    # zapisujemy przetwarzane dane
    ensure_dir(OUTPUT_PATH)
    with open(os.path.join(OUTPUT_PATH, 'train_text.pkl'), 'wb') as f:
        pickle.dump(train_text, f)
    with open(os.path.join(OUTPUT_PATH, 'train_summary.pkl'), 'wb') as f:
        pickle.dump(train_summary, f)
        
    # konwersja na liste numpy arrays
    train_text = [np.array(t, dtype=np.int64) for t in train_text]
    train_summary = [np.array(s, dtype=np.int64) for s in train_summary]
    
    batch_size = 32
    train_loader = create_dataloader(train_text, train_summary, batch_size=batch_size, shuffle=True)
    
    with open(VEC_FILE, 'rb') as f:
        glove_vectors = pickle.load(f)
    
    idx2word_enc = text_data.idx2word
    idx2word_dec = text_data.idx2word
    em_sz_enc = 100  
    em_sz_dec = 100 
    num_hidden = 256
    out_seq_length = 100
    num_layers = 2

    model = Seq2SeqAttention(glove_vectors, idx2word_enc, em_sz_enc, glove_vectors, idx2word_dec, em_sz_dec,
                             num_hidden, out_seq_length, num_layers=num_layers)
    model.cuda()

    fit(model, train_loader, learning_rate=1e-3, epochs=6, cycle_len=1, val_loader=None, 
        save=True, save_path=OUTPUT_PATH + '/checkpoint.pth.tar', print_period=100, grad_clip=0.0)
# zamiast 10 dalam 6 epok
if __name__ == "__main__":
    main()
