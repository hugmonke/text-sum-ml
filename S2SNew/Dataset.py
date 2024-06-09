from torch.utils.data import DataLoader, Dataset
import torch

class TextLoad(Dataset):
    def __init__(self, 
                 text: list, 
                 summary: list):
        self.text = torch.tensor(text)
        self.summary = torch.tensor(summary)
        
    def __len__(self): 
        return len(self.summary)
    
    def __getitem__(self, idx):
        return self.text[idx], self.summary[idx]
    
    @staticmethod
    def to_dataloader_(text: list, summary: list, batch_size: int=32, shuffle: bool=True):
        """ Returns an iterable data set. """
    
        return DataLoader(TextLoad(text, summary), batch_size=batch_size, shuffle=shuffle)