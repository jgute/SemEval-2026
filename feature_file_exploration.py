import string
import pandas as pd
from tabulate import tabulate


eng_datapath = "subtask1/train/eng.csv"

eng_dataset = pd.read_csv(eng_datapath)

spa_datapath = "subtask1/train/spa.csv"

spa_dataset = pd.read_csv(spa_datapath)

deu_datapath = "subtask1/train/deu.csv"

deu_dataset = pd.read_csv(deu_datapath)

def load_data(data):
    pol_dataset = data[data['polarization'] == 1]
    nonpol_dataset = data[data['polarization'] == 0]
    return pol_dataset, nonpol_dataset

def datasize(data):
    return len(data)

def average_text_length(data):
    return data['text'].str.len().mean()

def punctuation_counter(data):
   punctuation_count = data['text'].apply(lambda x: sum(ch in string.punctuation for ch in str(x))).sum()
   return punctuation_count

def punctuation_mean(data):
   punctuation_count = data['text'].apply(lambda x: sum(ch in string.punctuation for ch in str(x))).mean()
   return punctuation_count

heavydf = pd.DataFrame({
    'text': [
        "Hello!!! How are you??? I'm good!!!",
        "Wait... what?! Really?!??!!",
        "No way!!! This is crazy?!?!?!",
        "Wow!!! So many... punctuation??? Yes!!!",
        "##$$@@!!??--++==::;;,,...."
    ]
})

eng_pol, eng_nonpol = load_data(eng_dataset)
spa_pol, spa_nonpol = load_data(spa_dataset)
deu_pol, deu_nonpol = load_data(deu_dataset)

headers = ["Outputs", "English", "Spanish", "German"]
data = [['Polarizing length', datasize(eng_pol), datasize(spa_pol), datasize(deu_pol)],
        ['Nonpolarizing length',datasize(eng_nonpol), datasize(spa_nonpol), datasize(deu_nonpol)],
        ['Polarizing punctuation count', punctuation_counter(eng_pol), punctuation_counter(spa_pol), punctuation_counter(deu_pol)],
        ['Nonpolarizing punctuation count', punctuation_counter(eng_nonpol), punctuation_counter(spa_nonpol), punctuation_counter(deu_nonpol)],
        ['Polarizing average punctuation', punctuation_mean(eng_pol), punctuation_mean(spa_pol), punctuation_mean(deu_pol)],
        ['Nonpolarizing average punctuation', punctuation_mean(eng_nonpol), punctuation_mean(spa_nonpol), punctuation_mean(deu_nonpol)]]

print(tabulate(data, headers=headers, tablefmt="grid"))