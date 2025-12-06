import string
import pandas as pd



datapath = "subtask1/train/eng.csv"

dataset = pd.read_csv(datapath)

#load data into separate np arrays based on corresponding conditions
pol_data = dataset[dataset['polarization'] == 1]
nonpol_data = dataset[dataset['polarization'] == 0]
pol_data_count = len(pol_data)
nonpol_data_count = len(nonpol_data)

all_texts = dataset['text'].values
all_labels = dataset[('polarization')].values

average_polarizing_length = pol_data['text'].str.len().mean()
average_nonpol_length = nonpol_data['text'].str.len().mean()



print(f"Average polarizing length is {average_polarizing_length}")
print(f"Average nonpolarizing length is {average_nonpol_length}")



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

print(f"Size of polarizing entries is {pol_data_count}")
print(f"Size of nonpolarizing entries is {nonpol_data_count}")

print(f"Test for punctuation is {punctuation_counter(heavydf)}")
print(f"Count of polarizing punctuation is {punctuation_counter(pol_data)}")
print(f"Count of nonpolarizing punctuation is {punctuation_counter(nonpol_data)}")

print(f"Average of polarizing punctuation is {punctuation_mean(pol_data)}")
print(f"Average of nonpolarizing punctuation is {punctuation_mean(nonpol_data)}")