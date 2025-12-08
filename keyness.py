import math
import pandas as pd

datapath = "subtask1/train/eng.csv"

dataset = pd.read_csv(datapath)

# load data into separate np arrays based on corresponding conditions
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

# adapted from https://kristopherkyle.github.io/corpus-analysis-python/Python_Tutorial_7.html
def corpus_freq(text):
    freq = {}
    for word in text:
        # the first time we see a particular word we create a key:value pair
        if word not in freq:
            freq[word] = 1
        # when we see a word subsequent times, we add (+=) one to the frequency count
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
    """
    Takes a string of text and returns a list of the whitespace-separated and lowercased tokens.
    """

    tokens = text.lower().split()

    return tokens

pol_freq = corpus_freq(preprocess(full_pol_text))
nonpol_freq = corpus_freq(preprocess(full_nonpol_text))

print(pol_freq)
print(nonpol_freq)

keynesses = keyness(pol_freq,nonpol_freq)

print(keynesses['log-ratio'])

# given a list of words, return average log-ratio of all words
def get_keyness(text):
    sum = 0.0
    for word in text:
        if word in keynesses['log-ratio'].keys():
            sum += keynesses['log-ratio'][word]
    return sum / len(text)

print(get_keyness(['agreeing']))

