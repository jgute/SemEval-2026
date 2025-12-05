import numpy as np
import random
import torch
from typing import List
import pandas as pd


datapath = '/home/jay/PycharmProjects/SemEval-2026/subtask1/train/eng.csv'

dataset = pd.read_csv(datapath)

#load data into separate np arrays based on corresponding conditions
pol_data = dataset[dataset['polarization'] == 1]
nonpol_data = dataset[dataset['polarization'] == 0]

#validate data load
print(pol_data['text'].values[0])
print(nonpol_data['text'].values[0])

print(f"There are {len(pol_data)} positive labels.")
print(f"There are {len(nonpol_data)} negative labels.")
print(pol_data)



def split_dataset(texts, labels):
    tr_texts, de_texts, tr_labels, de_labels = train_test_split(texts, labels, test_size=.2, random_state=42)
    return tr_texts, de_texts, tr_labels, de_labels


