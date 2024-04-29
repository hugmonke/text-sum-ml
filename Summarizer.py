import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
block_size = 128
max_iters = 21
learning_rate = 1e-4
eval_iters = 20

#n_embd - Wymiar wektorow i ukrytych stanow
n_embd = 384
#n_head - ilosc "heads" - mechanizmow/warstw wykonujacych okreslone zadanie
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
        self.ln_f = nn.LayerNorm(n_embd)
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

            # Znajdz predykcje
            logits, loss = self.forward(index, device)
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



def get_random_chunk(split, encode):
    # Memory map dla malych wycinkow tekstu z pojedynczego pliku o dowolnym rozmiarze
    filename = "train_split.txt" if split == 'train' else "validate_split.txt"

    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Ustal rozmiar pliku i losowa pozycje startowa do czytania
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Szukaj losowej pozycji i sczytaj blok tekstu
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Dekoduj blok na string, ignoruj bledne sekwencje bajtow
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            
            # Trenuj i testuj odcinki (splits)
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
            
    return data

def get_batch(split, encode):
    data = get_random_chunk(split, encode)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters, encode):
    out = {}
    model.eval()
    for split in ['train', 'validate']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, encode)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():    
    print("Starting...")
    with open("vocab.txt", 'r', encoding='utf-8') as voca:
        vocabulary = voca.read()
        chars = sorted(list(set(vocabulary)))
            
    vocab_size = len(chars)
    
    string_int_dict = { ch:i for i,ch in enumerate(chars) }
    int_string_dict = { i:ch for i,ch in enumerate(chars) }
    encode = lambda string: [string_int_dict[char] for char in string]
    #decode = lambda num: ''.join([int_string_dict[i] for i in num])
    
    model = LanguageModel(vocab_size)
    print(f"Using: {device}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    while True:
        try:
            print('Loading existing model parameters...')
            with open('model-txt-sum.pkl', 'rb') as voca:
                model = pickle.load(voca)
            print('Loaded successfully!')
            m = model.to(device)
        except FileNotFoundError:
            print("Model does not exist yet!")
            print("Creating new model...")
            break
        else:
            break

    for iter in range(max_iters):
        print(iter)
        if iter % eval_iters == 0:
            print("Calculating loss...")
            losses = estimate_loss(model, max_iters, encode)
            print(f"Iteration: {iter}, Training loss: {losses['train']:.3f}, Validation loss: {losses['validate']:.3f}")
            

        xb, yb = get_batch('train', encode)

        #Wylicz loss
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    with open('model-txt-sum.pkl', 'wb') as vocab:
        pickle.dump(model, vocab)
    print('END: Model saved successfully!')
    
if __name__ == '__main__':
    main()
