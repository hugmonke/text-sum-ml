import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
jeden = torch.tensor([[1., -1.], [1., -1.], [1., -1.], [1., -1.], [1., -1.], [1., -1.]])
dwa = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
import spacy
from collections import Counter, defaultdict
from spacy.symbols import ORTH
text = "Gowno pies dupa cipa cyce"
freq = Counter(p for sent in text for p in sent)
print(freq.most_common(10000))
essa = [sent for sent in text]
dupsko = [p for p in essa]
print(dupsko)
print(Counter(dupsko))
print(type(np.array([1,2,3])))

idx2word = [letter for letter, count in freq.most_common(10000) if count > 1]
word2idx = defaultdict(lambda: "nigga", {letter: i for i, letter in enumerate(idx2word)})
print(idx2word)
print(word2idx)