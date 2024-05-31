import torch
import numpy as np
from preprocessing import TextDataSet
from model import Seq2SeqWithAttention
import pickle

def gimme_trained_model(path_m, glove_vec_path, idx2word_enc, idx2word_dec, emb_arg_enc, emb_arg_dec, num_hidden, out_seq_length, num_layers):

    with open(glove_vec_path, 'rb') as glove_pkl:
        global_vectors = pickle.load(glove_pkl)
    
    model = Seq2SeqWithAttention(global_vectors, idx2word_enc, emb_arg_enc, 
                                 global_vectors, idx2word_dec, emb_arg_dec,
                                 num_hidden, out_seq_length, num_layers=num_layers)
    
    model.cpu()

    checkpoint = torch.load(path_m)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def prepare_text(text, text_data):

    # tokenizuje i konwertuje na idx
    text_transformed = text_data.transform([text], tokenize=True)
    text_padded = np.array([np.array(t, dtype=np.int64) for t in text_transformed])
    return torch.tensor(text_padded).transpose(0, 1).cpu()

def decode_output(output, idx2word):

    # zamiana idx na slowa
    words = [idx2word[idx] for idx in output]
    return ' '.join(words)

if __name__ == "__main__":
    MODEL_PATH = "ProcessedData/checkpoint.pth.tar"
    GLOVE_PATH = "glove_vectors.pkl"
    
    em_sz_enc = 100  
    em_sz_dec = 100 
    num_hidden = 256
    out_seq_length = 100
    num_layers = 2

    text_data = TextDataSet(max_vocab=10000, maxlen=100)
    
    # zakoladam że idx2word_enc i idx2word_dec są takie same
    idx2word_enc = idx2word_dec = text_data.idx2word

    # ładuję model
    model = gimme_trained_model(MODEL_PATH, GLOVE_PATH, idx2word_enc, idx2word_dec, em_sz_enc, em_sz_dec, num_hidden, out_seq_length, num_layers)
    
    # tekst to strzeszczenia
    text = "Cześć kocham pieski. Mam na imię Natalia."
    
    # przygotowuje dane wejsciowe
    input_tensor = prepare_text(text, text_data)
    
    # no i generuję te strrreszvcze-mp eoweajndshu
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    # wyniki na tekst
    summary_indices = output.argmax(dim=-1).squeeze().tolist()
    summary = decode_output(summary_indices, idx2word_dec)
    
    print("Streszczenie:\n", summary)
