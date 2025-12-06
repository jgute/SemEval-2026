import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

datapath = "subtask1/train/eng.csv"

dataset = pd.read_csv(datapath)

#load data into separate np arrays based on corresponding conditions
pol_data = dataset[dataset['polarization'] == 1]
nonpol_data = dataset[dataset['polarization'] == 0]
pol_data_count = len(pol_data)
nonpol_data_count = len(nonpol_data)

all_texts = dataset['text'].values
all_labels = dataset[('polarization')].values

pol_texts = pol_data['text'].values
nonpol_texts = nonpol_data['text'].values

full_pol_text = pol_data['text'].str.cat(sep=" ")
full_nonpol_text = nonpol_data['text'].str.cat(sep=" ")


tfidf = TfidfVectorizer(stop_words='english')
result = tfidf.fit_transform([full_pol_text, full_nonpol_text])

print('\nWord indexes:')
print(tfidf.vocabulary_)
print('\ntf-idf value:')
print(result)
print('\ntf-idf values in matrix form:')
print(result.toarray())

data = result.data
indices = result.indices    # column indices
indptr = result.indptr      # row pointer

# find the indices in `data` corresponding to the 5 largest values
top5_idx = np.argsort(data)[:30]   # indices into the data array
top5_vals = data[top5_idx]

top5 = []

for idx in top5_idx:
    # find row number using `indptr`
    row = np.searchsorted(indptr, idx, side="right") - 1
    col = indices[idx]
    val = data[idx]
    top5.append((row, col, val))

feature_names = tfidf.get_feature_names_out()

top5_words = [
    (row, col, val, feature_names[col])
    for (row, col, val) in top5
]

for row, col, val, word in sorted(top5_words, key=lambda x: -x[2]):
    print(f"Row {row}, Word '{word}', TF-IDF={val:.5f}")

print(result.toarray())

top5_idx = np.argsort(pol_texts)[-30:]
top5_vals = pol_texts[top5_idx]

print(top5_vals)