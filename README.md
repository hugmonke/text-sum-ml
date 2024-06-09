# Text Summarisation AI - WIP

Projekt na seminarium "Matematyka dla przemysłu". Celem projektu jest stworzenie algorytmu programowania neurolingwistycznego, który na podstawie zadanego mu tekstu jest w stanie podać znajdujące się w tekście najważniejsze informacje.
AI opierać się będzie na bibliotece PyTorch.

Najważniejsze w projekcie jest dla nas dogłębne zrozumienie problemu i zapoznanie się z technikami NLP oraz poznanie bilioteki PyTorch.
W miarę możliwości chcemy osiągnąć możliwie najlepsze zrozumienie mechanizmów i metod matematycznych na których opiera się NLP.

# Info
The text summariser uses a Sequence to Sequence algorithm with attention. 

To train your model on the data given in requirements, use Main.py  
To generate examples out of data, on which the model was trained, use GenerateExamples.py  
To summarize your own text, use Summarizer.py  
Remember about pre-trained global vectors!

FH, NB, WO, JJ
# Requirements
Required libraries are to be found in lib_requirements.txt    
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/data    
https://www.kaggle.com/datasets/sawarn69/glove6b100dtxt  

A virtual environment is recommended.


# Sources & Literature:  
"Harrison Kinsley & Daniel Kukieła: Neural Networks From Scratch in Python" https://nnfs.io/  
https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html  
https://arxiv.org/pdf/1409.0473  
