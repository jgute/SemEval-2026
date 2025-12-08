import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk

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
full_text = full_pol_text + " " + full_nonpol_text

pol_data_copy = pol_data.copy()
pol_data_copy.loc[:, 'tokens'] = pol_data_copy['text'].map(lambda x: nltk.word_tokenize(x.lower()))
print("tokenized", print(pol_data_copy[:5]))


tfidf = TfidfVectorizer(stop_words='english')

pol_input = [full_pol_text, full_text]
nonpol_input = [full_nonpol_text, full_text]

def get_top_tfidf_words(text):
    result = tfidf.fit_transform(text)
    # Row 0 is full_pol_text
    pol_vector = result.toarray()[0]

    # Get feature names (words)
    words = np.array(tfidf.get_feature_names_out())

    # Get indices of top 10 values
    top_n = 100
    top_indices = pol_vector.argsort()[::-1][:top_n]

    # Extract words + scores
    top_words = words[top_indices]
    top_scores = pol_vector[top_indices]

    return top_words

    # print("\nTop 10 TF-IDF words:\n")
    # for word, score in zip(top_words, top_scores):
    #     print(f"{word}: {score:.6f}")


# print('\nWord indexes:')
# print(tfidf.vocabulary_)
# print('\ntf-idf value:')
# print(result)
# print('\ntf-idf values in matrix form:')
# print(result.toarray())

pol_top_tfidf = get_top_tfidf_words(pol_input)
nonpol_top_tfidf = get_top_tfidf_words(nonpol_input)