import string
import pandas as pd
from tabulate import tabulate
import emoji



eng_datapath = "subtask1/train/eng_new.csv"

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

def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = text.split()
    return len(tokens)

def average_words(data):
    average = data['text'].apply(preprocess).sum()
    return average / len(data)


def average_datasize(data):
    return (datasize(data) / len(data))

def average_text_length(data):
    return data['text'].str.len().mean()

def punctuation_counter(data):
    punctuation_count = data['text'].apply(lambda x: sum(ch in string.punctuation for ch in str(x))).sum()
    return punctuation_count

def punctuation_mean(data):
    average = punctuation_counter(data) / datasize(data)
    return average

def emoji_counter(data):
    emoji_count = data['text'].apply(emoji.emoji_count).sum()
    return emoji_count / len(data)

def emoji_sum(data):
    return data['text'].apply(emoji.emoji_count).sum()

text_with_emojis = "👋😊 This is a test. 🚀✨"
emoji_count = emoji.emoji_count(text_with_emojis)
print(f"emoji count is {emoji_count}")

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
data = [['Polarizing dataset size', datasize(eng_pol), datasize(spa_pol), datasize(deu_pol)],
        ['Nonpolarizing dataset size',datasize(eng_nonpol), datasize(spa_nonpol), datasize(deu_nonpol)],
        ['Polarizing average length', average_words(eng_pol), average_words(spa_pol), average_words(deu_pol)],
        ['Nonpolarizing average length', average_words(eng_nonpol), average_words(spa_nonpol), average_words(deu_nonpol)],
        ['Polarizing average punctuation', punctuation_mean(eng_pol), punctuation_mean(spa_pol), punctuation_mean(deu_pol)],
        ['Nonpolarizing average punctuation', punctuation_mean(eng_nonpol), punctuation_mean(spa_nonpol), punctuation_mean(deu_nonpol)],
        ['Polarizing average emoji count', emoji_sum(eng_pol), emoji_sum(spa_pol), emoji_sum(deu_pol)],
        ['Nonpolarizing average emoji count', emoji_sum(eng_nonpol), emoji_sum(spa_nonpol), emoji_sum(deu_nonpol)],
        ['Polarizing average emoji count', emoji_counter(eng_pol), emoji_counter(spa_pol), emoji_counter(deu_pol)],
        ['Nonpolarizing average emoji count', emoji_counter(eng_nonpol), emoji_counter(spa_nonpol), emoji_counter(deu_nonpol)]]

print(tabulate(data, headers=headers, tablefmt="grid"))