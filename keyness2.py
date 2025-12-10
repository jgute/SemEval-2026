import math
import pandas as pd
import numpy as np

datapath = "subtask2/train/eng.csv"

dataset = pd.read_csv(datapath)

#load data into separate pandas dataframes based on corresponding conditions
political_data = dataset[dataset['political'] == 1]
nonpolitical_data = dataset[dataset['political'] == 0]
racial_data = dataset[dataset['racial/ethnic'] == 1]
nonracial_data = dataset[dataset['racial/ethnic'] == 0]
religious_data = dataset[dataset['religious'] == 1]
nonreligious_data = dataset[dataset['religious'] == 0]
gender_data = dataset[dataset['gender/sexual'] == 1]
nongender_data = dataset[dataset['gender/sexual'] == 0]
other_data = dataset[dataset['other'] == 1]
nonother_data = dataset[dataset['other'] == 0]

political_labels = dataset['political'].values
racial_labels = dataset['racial/ethnic'].values
religious_labels = dataset['religious'].values
gender_labels = dataset['gender/sexual'].values
other_labels = dataset['other'].values

all_texts = dataset['text'].values
all_labels = np.vstack([political_labels, racial_labels, religious_labels, gender_labels, other_labels]).T

# adapted from https://kristopherkyle.github.io/corpus-analysis-python/Python_Tutorial_7.html
def corpus_freq(text):
    freq = {}
    for word in text:
        if word not in freq:
            freq[word] = 1
        else:
            freq[word] += 1
    return freq

# adapted from https://kristopherkyle.github.io/corpus-analysis-python/Python_Tutorial_7.html
def keyness(freq_dict1,freq_dict2):
    keyness_dict = {"log-ratio": {},"%diff" : {},"odds-ratio" : {}, "c1_only" : {}, "c2_only":{}}

    size1 = sum(freq_dict1.values())
    size2 = sum(freq_dict2.values())

    def log_ratio(freq1,size1,freq2,size2):
        freq1_norm = freq1/size1 * 1000000
        freq2_norm = freq2/size2 * 1000000
        index = math.log2(freq1_norm/freq2_norm)
        return(index)

    def perc_diff(freq1,size1,freq2,size2):
        freq1_norm = freq1/size1 * 1000000
        freq2_norm = freq2/size2 * 1000000
        index = ((freq1_norm-freq2_norm) * 100)/freq2_norm
        return(index)

    def odds_ratio(freq1,size1,freq2,size2):
        index = (freq1/(size1-freq1))/(freq2/(size2-freq2))
        return(index)

    all_words = set(list(freq_dict1.keys()) + list(freq_dict2.keys()))

    for item in all_words:
        if item not in freq_dict1:
            keyness_dict["c2_only"][item] = freq_dict2[item]/size2 * 1000000
            continue
        if item not in freq_dict2:
            keyness_dict["c1_only"][item] = freq_dict1[item]/size1 * 1000000
            continue

        keyness_dict["log-ratio"][item] = log_ratio(freq_dict1[item],size1,freq_dict2[item],size2)

        keyness_dict["%diff"][item] = perc_diff(freq_dict1[item],size1,freq_dict2[item],size2)

        keyness_dict["odds-ratio"][item] = odds_ratio(freq_dict1[item],size1,freq_dict2[item],size2)

    return keyness_dict

def preprocess(text):
    tokens = text.lower().split()

    return tokens

pol_freq = corpus_freq(preprocess(political_data['text'].str.cat(sep=" ")))
nonpol_freq = corpus_freq(preprocess(nonpolitical_data['text'].str.cat(sep=" ")))
racial_freq = corpus_freq(preprocess(racial_data['text'].str.cat(sep=" ")))
nonracial_freq = corpus_freq(preprocess(nonracial_data['text'].str.cat(sep=" ")))
religious_freq = corpus_freq(preprocess(religious_data['text'].str.cat(sep=" ")))
nonreligious_freq = corpus_freq(preprocess(nonreligious_data['text'].str.cat(sep=" ")))
gender_freq = corpus_freq(preprocess(gender_data['text'].str.cat(sep=" ")))
nongender_freq = corpus_freq(preprocess(nongender_data['text'].str.cat(sep=" ")))
other_freq = corpus_freq(preprocess(other_data['text'].str.cat(sep=" ")))
nonother_freq = corpus_freq(preprocess(nonother_data['text'].str.cat(sep=" ")))

pol_keynesses = keyness(pol_freq,nonpol_freq)
racial_keynesses = keyness(racial_freq, nonreligious_freq)
religious_keynesses = keyness(religious_freq, nonreligious_freq)
gender_keynesses = keyness(gender_freq, nongender_freq)
other_keynesses = keyness(other_freq, nonother_freq)

def get_log_ratio(text, keynesses):
    sum = 0.0
    for word in text:
        if word in keynesses['log-ratio'].keys():
            sum += keynesses['log-ratio'][word]
    return sum / len(text)

def get_perc_diff(text, keynesses):
    sum = 0.0
    for word in text:
        if word in keynesses['%diff'].keys():
            sum += keynesses['%diff'][word]
    return sum / len(text)

def get_odds_ratio(text, keynesses):
    sum = 0.0
    for word in text:
        if word in keynesses['odds-ratio'].keys():
            sum += keynesses['odds-ratio'][word]
    return sum / len(text)

