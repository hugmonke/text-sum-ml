import numpy as np
from more_itertools import chunked
from math import ceil
from itertools import chain
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
import re
import spacy
from spacy.symbols import ORTH
import keras


def flattenlist(listoflists):
    # Changes list of lists into single list: [[A, B, C], [D, E, F]] -> [A, B, C, D, E, F]
    return list(chain.from_iterable(listoflists))

class Tokenizer:
    # re.compile compiles regular expression (regex) for future use
    def __init__(self, lang='en_core_web_sm'):
        self.tokenizer = spacy.load(lang)
        # Ignores The Line Break element " </br>  "  in reviews
        self.br_regex = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
        # Add these strings as special cases
        for w in ('eos', 'bos', 'unk'):
            self.tokenizer.tokenizer.add_special_case(w, [{ORTH: w}])
            
    def replace_br(self, X): 
        # Replaces one or many matches with a string
        return self.br_regex.sub("\n", X)
    
    def spacy_tokenizer(self, proc_sen):
        # Converts letters to lower, replaces <br> with newline, returns a list
        return [x.text for x in self.tokenizer.tokenizer(self.replace_br(proc_sen.lower())) if not x.is_space]
    
    def process_text(self, sentence):
        # Uses regex for substitution
        proc_sen = re.sub(r'([/#])', r' \1 ', sentence)
        proc_sen = re.sub(' {2,}', ' ', sentence)
        return self.spacy_tokenizer(proc_sen)
       
    @staticmethod
    def process_all(text):
        tokenizer_object = Tokenizer()
        return [tokenizer_object.process_text(sentence) for sentence in text]
    
    def transform(self, text):
        return Tokenizer.process_all(text)

    def cpu_flatlist(self, X):
        core_usage = (cpu_count()+1) // 2
        # Multiprocessing for faster text transformation 
        with Pool(core_usage) as pool:
            chunk_size = ceil(len(X)/core_usage)
            results = pool.map(Tokenizer.process_all, chunked(X, chunk_size), chunksize=1)
        return flattenlist(results)
    
class TextDataSet:
    def __init__(self, max_vocab: int, maxlen: int, min_freq: int=1, padding: str='pre') -> ... :
        """
        Args:
            max_vocab (int): Maximum length of vocabulary. Takes n most common letters.
            maxlen (int): Maximum length of all padded sequences.
            min_freq (int, optional): Minimum required frequency for a letter. Defaults to 1.
            padding (str, optional): Pad either before or after each sequence. Defaults to 'pre' (before).
        """
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.maxlen = maxlen
        self.padding = padding
        self.tokenizer = Tokenizer()
    
    def fit(self, text: str, tokenize: bool=True) -> str:
        if tokenize:
            text = self.tokenizer.cpu_flatlist(text)
        self.freq = Counter(p for sent in text for p in sent)
        # List of letters that show up more than once
        self.idx2word = [letter for letter, count in self.freq.most_common(self.max_vocab) if count > self.min_freq]
        self.idx2word.insert(0, 'unk')
        self.idx2word.insert(1, 'pad')
        self.idx2word.insert(2, 'bos')
        self.idx2word.insert(3, 'eos')
        self.word2idx = defaultdict(lambda: 0, {letter: i for i, letter in enumerate(self.idx2word)})
        self.pad_int = self.word2idx['pad']
        return text
        
    def fit_transform(self, text: str, tokenize: bool=True) -> np.array:
        text = self.fit(text, tokenize=tokenize)
        text_padded = self.internal_transform(text, tokenize=False)
        return np.array(text_padded, dtype=object)

    def internal_transform(self, text: str, tokenize: bool=True) -> np.array:
        if tokenize:       
            text = self.tokenizer.cpu_flatlist(text)
        text_ints = [[self.word2idx[i] for i in sent] for sent in text]
        text_padded = keras.utils.pad_sequences(text_ints, maxlen=self.maxlen, padding=self.padding, value=self.pad_int)
        return np.array(text_padded, dtype=object)
    
    def transform(self, text: str, tokenize: bool=True, word2idx=None, maxlen=None, padding=None) -> np.array:
        if tokenize:       
            text = self.tokenizer.cpu_flatlist(text)
        if word2idx:
            self.word2idx = word2idx
        if maxlen:
            self.maxlen = maxlen
        if padding:
            self.padding = padding
        text_ints = [[self.word2idx[i] for i in sent] for sent in text]
        text_padded = keras.utils.pad_sequences(text_ints, maxlen=self.maxlen, padding=self.padding, value=self.pad_int)
        return np.array(text_padded, dtype=object)
    
# def limit_unk_vocab(train_text: str, train_summary: str, all_text_model, max_unk_text: int=1, max_unk_summary: int=1) -> np.array:
#     train_text_reduced = []
#     train_summary_reduced = []

#     for txt, sumy in zip(train_text, train_summary):
#         unk_txt = len([x for x in txt if x == all_text_model.word2idx['unk']])
#         unk_sumy = len([x for x in sumy if x == all_text_model.word2idx['unk']])
#         if (unk_txt <= max_unk_text) and (unk_sumy <= max_unk_summary):
#             train_text_reduced.append(txt.tolist())
#             train_summary_reduced.append(sumy.tolist())
        
#     assert(len(train_text_reduced) == len(train_summary_reduced))
#     print('New text size: ', len(train_text_reduced))
#     print('New summary size: ', len(train_summary_reduced))

#     return np.array(train_text_reduced, dtype=object), np.array(train_summary_reduced, dtype=object)