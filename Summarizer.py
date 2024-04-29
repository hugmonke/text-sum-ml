import torch
import torch.nn as nn
from torch.nn import functional as F
#from TrainTextSum import LanguageModel, Head, MultiHeadAttention, TransBlock, FeedFoward
import mmap
import random
import pickle
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
block_size = 128
max_iters = 100
learning_rate = 1e-4
eval_iters = 100
n_embd = 384
n_head = 1
n_layer = 1
dropout = 0.2
chars = ""

class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        #super() pozwala na przejecie zmiennych z inicjalizacji klasy-rodzica (tutaj nn.Module) 
        
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # Oblicz wartosci uwagi ("affinities" - "podobienstwo, pokrewienstwo")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T) mnozenie macierzy
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # Dokonaj wazonej agregacji (laczenia w calosci) wartosci 
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
        # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        
        out = self.dropout(self.proj(out))
        return out
    

class FeedFoward(nn.Module):
    """Simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
            )

    def forward(self, x):
        return self.net(x)
    
class TransBlock(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: wymiar wektorow
        # n_head: ilosc glow (warstw obliczeniowych)
        super().__init__()
        
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x
    
class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        
        B, T = index.shape
        
        # idx i targets (B,T) sa tensorami liczb calkowitych
        tok_emb = self.token_embedding_table(index) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = tok_emb + pos_emb #(B,T,C)
        x = self.blocks(x) #(B,T,C)
        x = self.ln_f(x) #(B,T,C)
        logits = self.lm_head(x) #(B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # Index to lista (B, T) indeksow w aktualnym kontekscie
        for _ in range(max_new_tokens):
            
            index_cond = index[:, -block_size:]
            # Znajdz predykcje
            logits, loss = self.forward(index_cond)
            # Skup sie na ostatnim kroku czasowym - staje sie (B, C)
            logits = logits[:, -1, :] 
            # Uzyj softmax dla znalezienia prawdopodobienstw (softmax - znormalizowana funkcja wykladnicza) - (B, C)
            probs = F.softmax(logits, dim=-1)
            # Pobierz probke z rozkladu - (B, 1)
            index_next = torch.multinomial(probs, num_samples=1) 
            # Dodaj pobrane indeksy do biezacej sekwencji - (B, T+1)
            
            index = torch.cat((index, index_next), dim=1) 
            #torch.cat - Concatenates the given sequence of seq tensors in the given dimension. 
            # All tensors must either have the same shape (except in the concatenating dimension) or be a 1-D empty tensor with size (0,).
        return index
    
    
def main():  
    with open("vocab.txt", 'r', encoding='utf-8') as f:
            text = f.read()
            chars = sorted(list(set(text)))
            
    vocab_size = len(chars)

    string_int_dict = { ch:i for i,ch in enumerate(chars) }
    int_string_dict = { i:ch for i,ch in enumerate(chars) }
    encode = lambda string: [string_int_dict[char] for char in string]
    decode = lambda num: ''.join([int_string_dict[i] for i in num])

    model = LanguageModel(vocab_size)
    
    while True:
        try:
            print('Loading model parameters...')
            with open('model-txt-sum.pkl', 'rb') as f:
                model = pickle.load(f)
            print('Loaded successfully!')
            m = model.to(device)
        except FileNotFoundError:
            print("Model does not exist yet!")
            print("You need to train your model first...")
            print("Use TrainTextSum.py to train your model!")
            break
        else:
            prompt = input("Enter Prompt:\n")
            if prompt == "KONIEC":
                break
            context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
            generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
            print(f'Completion:\n{generated_chars}')
            print("Type KONIEC to close the program\n")
        
if __name__ == '__main__':
    main()
