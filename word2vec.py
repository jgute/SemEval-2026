import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
import gensim

datapath = "subtask1/train/eng.csv"

dataset = pd.read_csv(datapath)

#load data into separate np arrays based on corresponding conditions
pol_data = dataset[dataset['polarization'] == 1]
nonpol_data = dataset[dataset['polarization'] == 0]
pol_data_count = len(pol_data)
nonpol_data_count = len(nonpol_data)

all_texts = dataset['text'].values
all_labels = dataset['polarization'].values

pol_texts = pol_data['text'].values
nonpol_texts = nonpol_data['text'].values

full_pol_text = pol_data['text'].str.cat(sep=" ")
full_nonpol_text = nonpol_data['text'].str.cat(sep=" ")

pol_data_copy = pol_data.copy()
pol_data_copy.loc[:, 'tokens'] = pol_data_copy['text'].map(lambda x: nltk.word_tokenize(x.lower()))
print("tokenized", print(pol_data_copy[:5]))

w2v_model = gensim.models.Word2Vec(pol_data_copy['tokens'].tolist(), vector_size=100, window=5, min_count=1)
w2v_model.save("word2vec.model")

def get_vectors(text):
    vectors = []
    for word in text:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])
    return vectors

vector = w2v_model.wv['israel']  # get numpy vector of a word
sims = w2v_model.wv.most_similar('hamas', topn=10)  # get other similar words

print(sims)
print(w2v_model.wv.most_similar('the'))
text_in = "this is a polarizing statement"
print(get_vectors(text_in.split()))
#
# w2v_dict = dict(zip(model.wv.index_to_key, model.wv.vectors))
# print(w2v_dict)
