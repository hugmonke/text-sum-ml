import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random


def randn(*sizes): 
    """ Returns a normalized tensor filled with random numbers from a standard normal distribution as a Parameter of the model"""
    return nn.Parameter(torch.randn(sizes)/np.sqrt(sizes[0]))
def embeddings(vect: dict, idx2word: list, embedding_size: int, pad_idx: int=1) -> nn.Embedding:
    """
    Creates embedding layers out of given word vectors
    
    Args:
    vect (dict): Word dictionary
    idx2word (list) - index to word: Word list
    emb_size (int): Size of embedding
    pad_idx (int): Padding index
    
    Returns:
    nn.Embedding: Embedding layer
    
    """
    # nn.Embedding is a lookup table that stores embeddings of a fixed dictionary and size
    embedding = nn.Embedding(num_embeddings=len(idx2word), embedding_dim=embedding_size, padding_idx=pad_idx)
    weights = embedding.weight.data
    for i, word in enumerate(idx2word):
        try:
            # Creates a tensor from a numpy array | MUST BE MULTIPLIED OR ELSE PADDING HAS TOO BIG WEIGHTS
            weights[i] = torch.from_numpy(vect[word] * 5)
        except KeyError:
            pass
    embedding.weight.requires_grad = False
    return embedding

class Seq2SeqWithAttention(nn.Module):
    
    def __init__(self, 
                num_hid_layers,
                vect_encoder, 
                vect_decoder,
                out_seq_len,
                num_layers, 
                idx2word, 
                emb_size, 
                pad_idx,  
                activation = F.tanh):
        
        super().__init__()
        self.num_hid_layers = num_hid_layers
        self.out_seq_len = out_seq_len
        self.num_layers = num_layers
        self.activation = activation
        
        # Encoder 
        self.encoder_embeddings = embeddings(vect=vect_encoder, idx2word=idx2word, embedding_size=emb_size, pad_idx=pad_idx)
        self.encoder_gru = nn.GRU(emb_size, num_hid_layers, num_layers, bidirectional=True) # See: cheatsheet 1
        self.encoder_out = nn.Linear(2*num_hid_layers, emb_size, bias=False)
        
        # nn.Dropout - During training, randomly zeroes some of the elements of the input tensor with probability p (p=0.1)
        # This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons as described in the paper: 
        # "Improving neural networks by preventing co-adaptation of feature detectors."
        self.encoder_dropout, self.encoder_dropout_emb = nn.Dropout(0.1), nn.Dropout(0.1)
        
        # Decoder params
        self.decoder_embeddings = embeddings(vect=vect_decoder, idx2word=idx2word, embedding_size=emb_size, pad_idx=pad_idx)
        self.decoder_gru = nn.GRU(emb_size, emb_size, num_layers=num_layers)
        self.decoder_out = nn.Linear(num_hid_layers, len(idx2word))
        self.decoder_dropout = nn.Dropout(0.1)
        self.decoder_out.weight = self.decoder_embeddings.weight
        
        # Attention params
        self.weight_matrix = randn(2*num_hid_layers, emb_size)
        self.value_matrix = randn(emb_size)   
        self.l2 = nn.Linear(2*num_hid_layers + emb_size, emb_size)
        self.l3 = nn.Linear(emb_size, emb_size)
        
    def forward(self,
                X: list,
                Y: list=None,
                tr_ratio: float = 0.0,
                return_attention: bool = False,
                ) -> torch.Tensor:
        
        # ENCODE FORWARD
        seq_len,  batch_size = X.size()
        hid_layers = torch.zeros(2*self.num_layers, batch_size, self.num_hid_layers).cpu()  
        encode_embeddings = self.encoder_dropout_emb(self.encoder_embeddings(X)) # with p=0.1 zeroes some of the elements of the encoder_embeddings
        encode_output, hid_layers = self.encoder_gru(encode_embeddings, hid_layers)
        # Changes shape, order, makes a copy as it was made from scratch, changes shape again
        hid_layers = hid_layers.view(2, 2, batch_size, -1).permute(0, 2, 1, 3).contiguous().view(2, batch_size, -1)
        
        # Applies dropout, creates a linear layer
        # This line multiplies two matrices and for that the above transformation is required
        hid_layers = self.encoder_out(self.encoder_dropout(hid_layers))
        
        # DECODE FORWARD
        # long() increases data type scope
        decode_input = torch.zeros(batch_size).long().cpu()
        att_enc = torch.matmul(encode_output, self.weight_matrix)
        
        results = []
        attentions = []
        
        for i in range(self.out_seq_len):
            att_dec = self.l3(hid_layers[-1]) #linear transformation Ax = B, x = hid_layers[-1]
            activ = self.activation(att_enc + att_dec)
            sftmx = F.softmax(torch.matmul(activ, self.value_matrix), dim=0)
            attentions.append(sftmx)
            Xa = (sftmx.unsqueeze(2) * encode_output).sum(0)
            dec_embs = self.decoder_embeddings(decode_input)
            weight_enc = self.l2(torch.cat([dec_embs, Xa], dim=1))
            outp, hid_layers = self.decoder_gru(weight_enc.unsqueeze(0), hid_layers)
            outp = self.decoder_out(self.decoder_dropout(outp[0]))
            results.append(outp)
            
            # Teacher forcing
            decode_input = outp.data.max(1)[1].cpu()
            if (decode_input == 1).all():
                break
            if (Y is not None) and (random.random() < tr_ratio):
                if i >= len(Y): 
                    break
                # Assign next value to decoder input
                decode_input = Y[i]
                
        if return_attention:
            return torch.stack(results), torch.stack(attentions)
        return torch.stack(results)