import pandas as pd
import os
import pickle
from torch.utils.data import DataLoader, Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, text, summary, transpose=False):
        self.transpose = transpose
        self.text = [torch.tensor(t, dtype=torch.long) for t in text]
        self.summary = [torch.tensor(s, dtype=torch.long) for s in summary]
        
    def __getitem__(self, idx):
        if self.transpose:
            return self.text[idx].T, self.summary[idx].T
        else:
            return self.text[idx], self.summary[idx]
    
    def __len__(self): 
        return len(self.summary)

def create_dataloader(text: list, summary: list, batch_size: int=32, shuffle: bool=True, transpose: bool=False) -> DataLoader:
    """
    Creates an iterable data set.

    Args:
        text (list): Text list
        summary (list): Summary list
        batch_size (int, optional): Batch size. Defaults to 32.
        shuffle (bool, optional): Boolean to decide whether to shuffle. Defaults to True.
        transpose (bool, optional): Boolean to decide whether to transpose. Defaults to False.

    Returns:
        DataLoader: Returns an iterable around the given Dataset
    """
    dataset = TextDataset(text, summary, transpose)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
