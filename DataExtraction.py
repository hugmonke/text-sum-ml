import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import concurrent.futures

    
def process_file(args):
    """Appends .txt files to output file, returns characters used in vocabulary """
    
    input_path, filename, output_file, vocab = args
    file_path = os.path.join(input_path, filename)
    with open(file_path, "rt", encoding="utf-8") as infile:
        text = infile.read()
        
    with open(output_file, "a", encoding="utf-8") as outfile:
        #Dodaje do jednego wielkiego tekstu pomniejsze teksty 
        outfile.write(text)
        
    #Zwraca znaki zawarte w tekscie by pozniej dodac je do slownika   
    characters = set(text)
    return characters

def txt_files_in_dir(directory):
    """Returns list of all files in directory ending with .txt"""
    
    #['name1.txt', 'name2.txt', ...]
    return [file for file in os.listdir(directory) if file.endswith(".txt")]


def process_files_in_parallel(files, input_file_path, output_file_path):
    """Parallel process for quicker results"""
    vocab = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        #concurrent.futures.ProcessPoolExecutor to klasa sluzaca do wykonywania operacji asynchronicznie
        #Dla kazdego zapytania odpala osobny proces
        #Z dokumentacji:
        # The ProcessPoolExecutor class is an Executor subclass that uses a pool of processes to execute calls asynchronously. 
        # ProcessPoolExecutor uses the multiprocessing module, 
        # which allows it to side-step the Global Interpreter Lock but also means that only picklable objects can be executed and returned.
        
        args = [(input_file_path, filename, output_file_path, vocab) for filename in files]
        #[(sciezka/folder, nazwa_pliku np. 001.txt, slownik)]

        
        #executor.map(funkcja, argumenty/iterables ) zwraca iterator, ktory zastosowuje sie do kazdej funkcji dla kazdego argumentu iterowalnego
        #Tutaj mamy mape wszystkich procesow, do kazdego procesu zastosuj funkcje process file dla iterowalnych args 
        #tqdm zamienia postep wykonywania funkcji w ladny pasek ladowania, mozna to wyrzucic - bedzie dzialac bez tego
        for characters in tqdm(executor.map(process_file, args), total=len(files)):
            #Uruchomienie petli tworzy train_split.txt oraz validate_split.txt
            
            vocab.update(characters)
            #Oprocz tego odajemy znaki do slownika
            
    return vocab

def main():    
    
    input_train_path = "Test"
    input_validate_path = "Val"
    output_train_path = "train_split.txt"
    output_valiate_path = "validate_split.txt"
    vocab_file = "vocab.txt"
    while True:
        if os.path.isfile("train_split.txt") or os.path.isfile("validate_split.txt"):
            a = str(input("Do you want to overwrite existing files? (y/n)\n"))
            if a == 'y' or a == 'Y':
                print("Overwriting existing files...\n")
                break
            elif a == 'n' or a == 'N':
                print("Stopping data extraction\n")
                return 0
            else:
                print("Wrong input\n")
        else:
            break
            
    files_train = txt_files_in_dir(input_train_path)
    files_validate = txt_files_in_dir(input_validate_path)
    total_files_length = len(files_train) + len(files_validate)
    print("Total length of files is: ", total_files_length)

    # Tworzy puste pliki wyjsciowe, by potem dodawac do nich zdania
    open(output_train_path, 'w').close()
    open(output_valiate_path, 'w').close()

    # Przetwarzanie trenignowych i testowych danych
    vocab_train = process_files_in_parallel(files_train, input_train_path, output_train_path)
    vocab_validate = process_files_in_parallel(files_validate, input_validate_path, output_valiate_path)

    vocab = vocab_train.union(vocab_validate)
    #Suma wszystkich znalezionych uzytych znakow
    with open(vocab_file, "w", encoding="utf-8") as vfile:
        for char in sorted(vocab):
            vfile.write(f'{char}\n')
            #Wypisuje znaki w jednej kolumnie

if __name__ == '__main__':
    main()
    print("Done!\n")
