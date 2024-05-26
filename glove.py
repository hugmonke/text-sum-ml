
import numpy as np
import pickle

def load_glove_vectors(file_path):
    glove_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_vectors[word] = vector
    return glove_vectors

def save_glove_vectors(glove_vectors, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(glove_vectors, f)

glove_file_path = 'C:/Users/blasz/OneDrive/Pulpit/summarization_proj/glove.6B.100d.txt'
# Ścieżka do pliku pickle, w którym zapisane będą wektory
pickle_file_path = 'C:/Users/blasz/OneDrive/Pulpit/summarization_proj/glove_vectors.pkl'

glove_vectors = load_glove_vectors(glove_file_path)
save_glove_vectors(glove_vectors, pickle_file_path)