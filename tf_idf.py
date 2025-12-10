import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk

eng_datapath = "subtask1/train/eng.csv"
spa_datapath = "subtask1/train/spa.csv"
deu_datapath = "subtask1/train/deu.csv"

tfidf = TfidfVectorizer(stop_words='english')

def get_inputs(datapath):
    dataset = pd.read_csv(datapath)

    pol_data = dataset[dataset['polarization'] == 1]
    nonpol_data = dataset[dataset['polarization'] == 0]

    full_pol_text = pol_data['text'].str.cat(sep=" ")
    full_nonpol_text = nonpol_data['text'].str.cat(sep=" ")
    full_text = full_pol_text + " " + full_nonpol_text

    pol_data_copy = pol_data.copy()
    pol_data_copy.loc[:, 'tokens'] = pol_data_copy['text'].map(lambda x: nltk.word_tokenize(x.lower()))

    pol_input = [full_pol_text, full_text]
    nonpol_input = [full_nonpol_text, full_text]

    return pol_input, nonpol_input

def get_top_tfidf_words(text):
    result = tfidf.fit_transform(text)
    pol_vector = result.toarray()[0]

    words = np.array(tfidf.get_feature_names_out())

    top_n = 100
    top_indices = pol_vector.argsort()[::-1][:top_n]

    top_words = words[top_indices]

    return top_words

pol_eng, nonpol_eng = get_inputs(eng_datapath)
pol_spa, nonpol_spa = get_inputs(spa_datapath)
pol_deu, nonpol_deu = get_inputs(deu_datapath)

pol_top_tfidf_eng = get_top_tfidf_words(pol_eng)
nonpol_top_tfidf_eng = get_top_tfidf_words(nonpol_eng)

pol_top_tfidf_spa = get_top_tfidf_words(pol_spa)
nonpol_top_tfidf_spa = get_top_tfidf_words(nonpol_spa)

pol_top_tfidf_deu = get_top_tfidf_words(pol_deu)
nonpol_top_tfidf_deu = get_top_tfidf_words(nonpol_deu)