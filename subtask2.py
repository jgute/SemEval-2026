import math

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import word_count
import keyness2
from sklearn.metrics import f1_score as sklearn_f1

datapath = "subtask2/train/eng.csv"

dataset = pd.read_csv(datapath)

#load data into separate pandas dataframes based on corresponding conditions
political_data = dataset[dataset['political'] == 1]
racial_data = dataset[dataset['racial/ethnic'] == 1]
religious_data = dataset[dataset['religious'] == 1]
gender_data = dataset[dataset['gender/sexual'] == 1]
other_data = dataset[dataset['other'] == 1]

political_labels = dataset['political'].values
racial_labels = dataset['racial/ethnic'].values
religious_labels = dataset['religious'].values
gender_labels = dataset['gender/sexual'].values
other_labels = dataset['other'].values

all_texts = dataset['text'].values
all_labels = np.vstack([political_labels, racial_labels, religious_labels, gender_labels, other_labels]).T

num_features = 10

def split_dataset(texts, labels):
    tr_texts, de_texts, tr_labels, de_labels = train_test_split(texts, labels, test_size=.2, random_state=42)
    return tr_texts, de_texts, tr_labels, de_labels

train_texts, dev_texts, train_labels, dev_labels = split_dataset(all_texts, all_labels)

def preprocess(text):
    tokens = text.lower().split()

    return tokens

path_to_negatives = "negative-words.txt"
negative_words = []
high_freq_pol_words = word_count.pol_words
high_freq_nonpol_words = word_count.nonpol_words
with open(path_to_negatives, "r") as file:
    for line in file:
        text = line.rstrip()
        negative_words.append(text)

def get_negative_tokens(text):
    negative_tokens_found = [token for token in text if token in negative_words]
    return len(negative_tokens_found)

def get_word_count(text):
    return len(text)

def get_high_freq_pol_words(text):
    high_freq_pol_words_found = [token for token in text if token in high_freq_pol_words]
    return len(high_freq_pol_words_found)

def get_high_freq_nonpol_words(text):
    high_freq_nonpol_words_found = [token for token in text if token in high_freq_nonpol_words]
    return len(high_freq_nonpol_words_found)

def get_keyness_log_ratio(text, keynesses):
    return keyness2.get_log_ratio(text, keynesses)

def get_keyness_perc_diff(text, keynesses):
    return keyness2.get_perc_diff(text, keynesses)

def get_keyness_odds_ratio(text, keynesses):
    return keyness2.get_odds_ratio(text, keynesses)

def extract_features(text):
    features = []
    features.append(get_keyness_log_ratio(text, keyness2.pol_keynesses))
    features.append(get_keyness_log_ratio(text, keyness2.racial_keynesses))
    features.append(get_keyness_log_ratio(text, keyness2.religious_keynesses))
    features.append(get_keyness_log_ratio(text, keyness2.gender_keynesses))
    features.append(get_keyness_log_ratio(text, keyness2.other_keynesses))
    return features

def featurize_data(texts, labels):
    features = [
        extract_features(preprocess(text_in)) for text_in in texts
    ]
    return torch.FloatTensor(features), torch.FloatTensor(labels)


def standardize(features: torch.Tensor) -> torch.Tensor:
    num_columns = features.shape[1]
    means = []
    stds = []

    for col in range(num_columns):
        column = features[:, col]
        total = 0
        for row in range(len(column)):
            total += features[row][col]
        means.append(total / len(column))

    for col in range(num_columns):
        column = features[:, col]
        total = 0
        for row in range(len(column)):
            total += pow((features[row][col] - means[col]), 2)
        stds.append(math.sqrt(total / len(column)))

    for col in range(num_columns):
        column = features[:, col]
        for row in range(len(column)):
            features[row][col] = (column[row] - means[col]) / stds[col]

    return features

class MultilabelClassifier(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 5),
            torch.nn.Sigmoid()
        )
        self.output_size = 5
        self.coefficients = torch.nn.Linear(input_dim, self.output_size)
        initialize_weights(self.coefficients)

    def forward(self, features: torch.Tensor):
        return torch.sigmoid(self.coefficients(features))


def initialize_weights(coefficients):
    torch.nn.init.ones_(coefficients.weight)

    return coefficients


def logistic_loss(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.BCELoss()
    output = loss(prediction, label)

    return output

def make_optimizer(model, learning_rate) -> torch.optim:
    return torch.optim.SGD(model.parameters(), learning_rate)

def predict(model, features):
    with torch.no_grad():
        logits = model(features)
        return (logits > 0.5).int().numpy()  # shape [num_samples, num_classes]

def f1_score_multilabel(true_labels, predicted_labels):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    return sklearn_f1(true_labels, predicted_labels, average='samples', zero_division=0)

def training_loop(
        num_epochs,
        batch_size,
        train_features,
        train_labels,
        dev_features,
        dev_labels,
        optimizer,
        model
):
    samples = list(zip(train_features, train_labels))
    random.shuffle(samples)
    batches = []
    for i in range(0, len(samples), batch_size):
        batches.append(samples[i:i + batch_size])
    print("Training...")
    train_losses = []
    dev_losses = []
    for i in range(num_epochs):
        for batch in tqdm(batches):
            features, labels = zip(*batch)
            features = torch.stack(features)
            labels = torch.stack(labels)
            optimizer.zero_grad()
            logits = model(features)
            loss = logistic_loss(torch.squeeze(logits), labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            dev_logits = model(dev_features)
            dev_loss = logistic_loss(torch.squeeze(dev_logits), dev_labels)
            dev_losses.append(dev_loss.item())

        dev_f1 = f1_score_multilabel(dev_labels.tolist(), predict(model, dev_features))
        print(f"epoch {i}")
        print(f"Train loss: {sum(train_losses) / len(train_losses)}")
        print(f"Dev loss: {sum(dev_losses) / len(dev_losses)}")
        print(f"Dev F1 {dev_f1}")

    return model, train_losses, dev_losses


# run the model
train_features, train_labels_tensor = featurize_data(train_texts, train_labels)
train_features = standardize(train_features)
dev_features, dev_labels_tensor = featurize_data(dev_texts, dev_labels)
dev_features = standardize(dev_features)

num_features = 5
num_epochs = 20
model = MultilabelClassifier(input_dim=num_features)
learning_rate = 0.01
optimizer = make_optimizer(model, learning_rate)

trained_model, train_losses, dev_losses = training_loop(
    num_epochs,
    16,
    train_features,
    train_labels_tensor,
    dev_features,
    dev_labels_tensor,
    optimizer,
    model
)