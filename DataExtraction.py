import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import concurrent.futures

    
def process_file(args):
    directory, filename, output_file, vocab = args
    file_path = os.path.join(directory, filename)
    with open(file_path, "rt", encoding="utf-8") as infile:
        text = infile.read()
    with open(output_file, "a", encoding="utf-8") as outfile:
        outfile.write(text)
    characters = set(text)
    return characters

def txt_files_in_dir(directory):
    return [file for file in os.listdir(directory) if file.endswith(".txt")]

def process_files_in_parallel(files, folder_path, output_file):
    vocab = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        args = [(folder_path, filename, output_file, vocab) for filename in files]
        for characters in tqdm(executor.map(process_file, args), total=len(files)):
            vocab.update(characters)
            
    return vocab

def main():    
    
    folder_path_train = "Test"
    folder_path_validate = "Val"
    output_file_train = "train_split.txt"
    output_file_validate = "validate_split.txt"
    vocab_file = "vocab.txt"

    files_train = txt_files_in_dir(folder_path_train)
    files_validate = txt_files_in_dir(folder_path_validate)
    total_files = len(files_train) + len(files_validate)
    print(total_files)

    # Upewnia sie czy pliki wejsciowe sa puste zanim zostana dodane
    open(output_file_train, 'w').close()
    open(output_file_validate, 'w').close()

    # Przetwarzanie trenignowych i testowych danych
    vocab_train = process_files_in_parallel(files_train, folder_path_train, output_file_train)
    vocab_validate = process_files_in_parallel(files_validate, folder_path_validate, output_file_validate)

    vocab = vocab_train.union(vocab_validate)
    with open(vocab_file, "w", encoding="utf-8") as vfile:
        for char in sorted(vocab):
            vfile.write(char + '\n')

if __name__ == '__main__':
    main()