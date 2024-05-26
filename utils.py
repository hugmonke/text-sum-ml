import numpy as np
import pandas as pd
import os
import pickle


DATA_FILE = "C:/Users/blasz/OneDrive/Pulpit/summarization_proj/Reviews.csv"
OUTPUT_PATH = "C:/Users/blasz/OneDrive/Pulpit/summarization_proj/ExtractedData/"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
data = pd.read_csv(DATA_FILE)
articles = data['Text'].tolist()
summaries = data['Summary'].tolist()
for i in range(len(data)):
    if i % 200 == 0:
        print(f"Processed {i} records")
with open(os.path.join(OUTPUT_PATH, 'articles.pkl'), 'wb') as f:
    pickle.dump(articles, f)

with open(os.path.join(OUTPUT_PATH, 'summaries.pkl'), 'wb') as f:
    pickle.dump(summaries, f)

print("COMPLETED")