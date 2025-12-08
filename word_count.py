import string

import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

nltk.download('stopwords')

eng_datapath = "subtask1/train/eng.csv"
spa_datapath = "subtask1/train/spa.csv"
deu_datapath = "subtask1/train/deu.csv"

eng_dataset = pd.read_csv(eng_datapath)
spa_dataset = pd.read_csv(spa_datapath)
deu_dataset = pd.read_csv(deu_datapath)

#load data into separate np arrays based on corresponding conditions
pol_data = eng_dataset[eng_dataset['polarization'] == 1]
nonpol_data = eng_dataset[eng_dataset['polarization'] == 0]
pol_data_count = len(pol_data)
nonpol_data_count = len(nonpol_data)

all_texts = eng_dataset['text'].values
all_labels = eng_dataset[('polarization')].values

pol_texts = pol_data['text'].values
nonpol_texts = nonpol_data['text'].values

def get_pol_texts(dataset_in):
    pol_data = dataset_in[dataset_in['polarization'] == 1]
    pol_texts = pol_data['text'].values
    return pol_texts

def get_nonpol_texts(dataset_in):
    nonpol_data = dataset_in[dataset_in['polarization'] == 0]
    nonpol_texts = nonpol_data['text'].values
    return nonpol_texts

def most_common_words(text):
    word_counts = dict()
    words = [word.lower() for sentence in text for word in sentence.split()]

    # remove stopwords
    spa_stopwords = stopwords.words('spanish')
    eng_stopwords = stopwords.words('english')
    deu_stopwords = stopwords.words('german')
    all_stopwords = spa_stopwords + eng_stopwords + deu_stopwords
    words = [word for word in words if word not in all_stopwords]

    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

    word_counts = list(word_counts.items())

    n = 100

    top_n = sorted(word_counts, key=lambda item: item[1], reverse=True)[:n]
    return [word for word, count in top_n]

pol_words = most_common_words(get_pol_texts(eng_dataset))
nonpol_words = most_common_words(get_nonpol_texts(eng_dataset))
#
# print(most_common_words(get_pol_texts(eng_dataset)))
# print(most_common_words(get_pol_texts(spa_dataset)))
# print(most_common_words(get_pol_texts(deu_dataset)))
#
# print(most_common_words(get_nonpol_texts(eng_dataset)))
# print(most_common_words(get_nonpol_texts(spa_dataset)))
# print(most_common_words(get_nonpol_texts(deu_dataset)))