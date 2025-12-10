import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import nltk

nltk.download('punkt')


Data = pd.read_csv('subtask1/train/eng.csv')

def custom_nltk_tokenizer(text):
    return nltk.word_tokenize(text)

labels = Data['polarization']

train, dev = train_test_split(Data, test_size=.2, train_size=.8, stratify=labels,random_state=42)
count_v0 = CountVectorizer()
counts_all = count_v0.fit_transform(Data['text'])
count_v1 = CountVectorizer(vocabulary=count_v0.vocabulary_)
counts_train = count_v1.fit_transform(train['text'])
count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_)
counts_test = count_v2.fit_transform(dev['text'])
tfidftransformer = TfidfTransformer()
train_data = tfidftransformer.fit(counts_train).transform(counts_train)
test_data = tfidftransformer.fit(counts_test).transform(counts_test)

print(train)
print(dev)

x_train = train_data
x_test = test_data

y_train = (train['polarization'])
y_test = (dev['polarization'])

clf = MultinomialNB(alpha=0.01)
clf.fit(x_train, y_train)

pred_test = clf.predict_proba(x_test[10])
print(pred_test)

preds = clf.predict_proba(x_test)

p = []
gg = y_test.values
preds = preds.tolist()
for i in range(len(preds)):
    a = gg[i]
    b = preds[i]
    p.append(b[a])

y_true = np.array(y_test)
y_score = np.array(p)

predicted_classes = [1 if prob[1] >= 0.5 else 0 for prob in preds]

f1 = f1_score(y_test, predicted_classes)

print(f"F1 score is: {f1}")
