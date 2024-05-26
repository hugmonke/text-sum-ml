import pandas as pd
import os
import pickle
from torch.utils.data import DataLoader, Dataset

def load_data(file_path):
    """
    Wczytuje dane z pliku CSV.

    Parameters:
    file_path (str): Ścieżka do pliku CSV.

    Returns:
    DataFrame: Ramka danych zawierająca wczytane dane.
    """
    data = pd.read_csv(file_path)
    return data

def save_data(data, output_path, file_name):
    """
    Zapisuje dane do pliku CSV.

    Parameters:
    data (DataFrame): Ramka danych do zapisania.
    output_path (str): Ścieżka do folderu, w którym ma zostać zapisany plik.
    file_name (str): Nazwa pliku CSV do zapisania.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data.to_csv(os.path.join(output_path, file_name), index=False)

def save_pickle(data, output_path, file_name):
    """
    Zapisuje dane do pliku pickle.

    Parameters:
    data: Dane do zapisania.
    output_path (str): Ścieżka do folderu, w którym ma zostać zapisany plik.
    file_name (str): Nazwa pliku pickle do zapisania.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, file_name), 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    """
    Wczytuje dane z pliku pickle.

    Parameters:
    file_path (str): Ścieżka do pliku pickle.

    Returns:
    Dane wczytane z pliku pickle.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

from torch.utils.data import Dataset, DataLoader
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

def create_dataloader(text, summary, batch_size=32, shuffle=True, transpose=False):
    """
    Tworzy DataLoader dla danych tekstowych.

    Parameters:
    text (list): Lista tekstów.
    summary (list): Lista streszczeń.
    batch_size (int): Rozmiar batcha.
    shuffle (bool): Czy losowo przetasować dane.
    transpose (bool): Czy transponować dane.

    Returns:
    DataLoader: DataLoader z danymi.
    """
    dataset = TextDataset(text, summary, transpose)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
