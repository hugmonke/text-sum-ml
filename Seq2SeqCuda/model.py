import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random

# Auxiliary functions

def rand_t(*argt): 
    """ Returns a normalized tensor filled with random numbers from a standard normal distribution """
    return torch.randn(argt)/np.sqrt(argt[0])

def rand_p(*argp): 
    """ Returns the given tensor as a parameter of the module """
    return nn.Parameter(rand_t(*argp))

def create_embeddings(vect: dict, idx2word: list, embedding_size: int, pad_idx: int=1) -> nn.Embedding:
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
    
    # Embedding is a numerical representation of an object
    embedding = nn.Embedding(len(idx2word), embedding_size, padding_idx=pad_idx)
    weights = embedding.weight.data
    embedding.weight.requires_grad = False
    
    for i, word in enumerate(idx2word):
        try:
            # Creates a tensor from a numpy array
            #weights[i] = torch.from_numpy(vect[word] * 3)
            weights[i] = torch.from_numpy(vect[word])
        except KeyError:
            pass

    #embedding.weight.requires_grad = False
    return embedding



# Seq2Seq definition

class Seq2SeqWithAttention(nn.Module):
    """
    Seq2Seq - Sequence to sequence transformation algorithm
    Attention - A mechanism used to keep track of which parts of source input are important
    
    Args:
        Module super class 
    """
    # linear layer connect input and output neurons through linear transformation
    # encoder - reads source sequence and produces its representation
    # decoder - uses source representation from the encoder to generate the target sequence

    def __init__(self, vecs_enc, idx2word_enc, emb_arg_enc, 
                 vecs_dec, idx2word_dec, emb_arg_dec, 
                 num_hidden, outsequence_len, num_layers=2, 
                 activation=F.tanh, pad_idx=1):
        """
        Args:
            vecs_enc (list): Encoded vectors
            idx2word_enc (list): Index to word encoder input
            emb_arg_enc (list): The number of expected features in the input
            vecs_dec (list): Decoded vectors
            idx2word_dec (list): Index to word decoder input
            emb_arg_dec (int): Embedding argument decoder
            num_hidden (int): The number of features in the hidden state 
            outsequence_len (int): Length of output word sequence
            num_layers (int, optional): Number of recurrent layers. Default equal to 2 means stacking two GRUs together to form a stacked GRU. 
            activation (functional, optional): Neuron activation function. Defaults to F.tanh.
            pad_idx (int, optional): _description_. Defaults to 1.
        """
 
        super().__init__()
        self.activation = activation
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.out_seq_length = outsequence_len
        
        # Encoder params
        # Applies multi-layer gated recurrent unit (GRU)
        self.encoder_gru = nn.GRU(emb_arg_enc, num_hidden, num_layers=num_layers, bidirectional=True)
        self.encoder_embeddings = create_embeddings(vecs_enc, idx2word_enc, emb_arg_enc, pad_idx)
        # Creates a linear layer
        self.encoder_out = nn.Linear(num_hidden*2, emb_arg_dec, bias=False)
        
        # nn.Dropout - During training, randomly zeroes some of the elements of the input tensor with probability p (p=0.1)
        # This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons as described in the paper: 
        # "Improving neural networks by preventing co-adaptation of feature detectors."
        self.encoder_dropout, self.encoder_dropout_emb = nn.Dropout(0.1), nn.Dropout(0.1)
        
        # Decoder params
        self.decoder_gru = nn.GRU(emb_arg_dec, emb_arg_dec, num_layers=num_layers)
        self.decoder_embeddings = create_embeddings(vecs_dec, idx2word_dec, emb_arg_dec, pad_idx)
        self.out = nn.Linear(num_hidden, len(idx2word_dec))
        self.decoder_dropout = nn.Dropout(0.1)
        self.out.weight.data = self.decoder_embeddings.weight.data
        
        # Attention params
        self.weight_matrix = rand_p(2*num_hidden, emb_arg_dec)
        self.value_matrix = rand_p(emb_arg_dec)   
        self.l2 = nn.Linear(2*num_hidden + emb_arg_dec, emb_arg_dec)
        self.l3 = nn.Linear(emb_arg_dec, emb_arg_dec)
        
             

    def initHiddenCPU(self, batch_size: float) -> torch.Tensor:
        """ Initializes hidden layers using CPU"""
        return torch.zeros(2*self.num_layers, batch_size, self.num_hidden).cpu()    
    
    def initHiddenCUDA(self, batch_size: float) -> torch.Tensor:
        """ Initializes hidden layers using CUDA"""
        return torch.zeros(2*self.num_layers, batch_size, self.num_hidden).cuda()   
    
    def forward(self, 
                X: list,
                Y: list=None, 
                tf_ratio: float=0.0, 
                return_attention: bool=False
                ) -> torch.Tensor:
        
        # Encode forward
        seq_len, batch_size = X.size()
        hidden_layers = self.initHiddenCUDA(batch_size)
        encode_embeddings = self.encoder_dropout_emb(self.encoder_embeddings(X))
        encode_output, hidden_layers = self.encoder_gru(encode_embeddings, hidden_layers)
        
        # Changes shape, oder, makes a copy as it was made from scratch, changes shape again
        hidden_layers = hidden_layers.view(2, 2, batch_size, -1).permute(0, 2, 1, 3).contiguous().view(2, batch_size, -1)
        
        # Applies dropout, creates a linear layer
        hidden_layers = self.encoder_out(self.encoder_dropout(hidden_layers))
        
        # Decode forward
        # long() increases data type scope
        decode_input = torch.zeros(batch_size).long().cuda()
        #w1e = encode_output @ self.weight_matrix
        w1e = torch.matmul(encode_output, self.weight_matrix)
        
        results = []
        attentions = []
        
        for i in range(self.out_seq_length):
            w2d = self.l3(hidden_layers[-1])
            activ = self.activation(w1e + w2d)
            #a = F.softmax(u @ self.value_matrix, dim=0)
            
            sftmx = F.softmax(torch.matmul(activ, self.value_matrix), dim=0)
            attentions.append(sftmx)
            Xa = (sftmx.unsqueeze(2) * encode_output).sum(0)
            dec_embs = self.decoder_embeddings(decode_input)
            weight_enc = self.l2(torch.cat([dec_embs, Xa], dim=1))
            outp, hidden_layers = self.decoder_gru(weight_enc.unsqueeze(0), hidden_layers)
            outp = self.out(self.decoder_dropout(outp[0]))
            results.append(outp)
            
            # Teacher forcing
            decode_input = outp.data.max(1)[1].cuda()
            if (decode_input == 1).all():
                break
            if (Y is not None) and (random.random() < tf_ratio):
                if i >= len(Y): 
                    break
                # Assign next value to decoder input
                decode_input = Y[i]
                
        if return_attention:
            return torch.stack(results), torch.stack(attentions)
        return torch.stack(results)
        
        
    
