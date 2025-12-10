import torch
import pandas as pd
import math
import csv
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import word_count
from tf_idf import pol_top_tfidf_eng, nonpol_top_tfidf_eng
import keyness
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

datapath = "subtask1/train/eng.csv"
dataset = pd.read_csv(datapath)

pol_data = dataset[dataset['polarization'] == 1]
nonpol_data = dataset[dataset['polarization'] == 0]

all_texts = dataset['text'].values
all_labels = dataset['polarization'].values

def split_dataset(texts, labels):
    tr_texts, de_texts, tr_labels, de_labels = train_test_split(texts, labels, test_size=.2, random_state=42)
    return tr_texts, de_texts, tr_labels, de_labels

train_texts, dev_texts, train_labels, dev_labels = split_dataset(all_texts, all_labels)

def preprocess(text):
    tokens = text.lower().split()

    return tokens

path_to_positives = "positive-words.txt"
path_to_negatives = "negative-words.txt"
positive_words, negative_words = [], []
high_freq_pol_words = word_count.pol_words
high_freq_nonpol_words = word_count.nonpol_words

with open(path_to_negatives, "r") as file:
    for line in file:
        text = line.rstrip()
        negative_words.append(text)

with open(path_to_positives, "r") as file:
    for line in file:
        text = line.rstrip()
        positive_words.append(text)

def get_negative_tokens(text):
    negative_tokens_found = [token for token in text if token in negative_words]
    return len(negative_tokens_found)

def get_positive_tokens(text):
    positive_tokens_found = [token for token in text if token in positive_words]
    return len(positive_tokens_found)

def get_word_count(text):
    return len(text)

def get_high_freq_pol_words(text):
    high_freq_pol_words_found = [token for token in text if token in high_freq_pol_words]
    return len(high_freq_pol_words_found)

def get_high_freq_nonpol_words(text):
    high_freq_nonpol_words_found = [token for token in text if token in high_freq_nonpol_words]
    return len(high_freq_nonpol_words_found)

def get_keyness_log_ratio(text):
    return keyness.get_log_ratio(text)

def extract_features(text):
    features = []
    features.append(get_negative_tokens(text))
    features.append(get_positive_tokens(text))
    features.append(get_high_freq_pol_words(text))
    features.append(get_high_freq_nonpol_words(text))
    features.append(get_word_count(text))
    features.append(get_keyness_log_ratio(text))

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

class SentimentClassifier(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.output_size = 1
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
        predicted_labels = []

        for logit in logits:
            if logit > 0.5:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)

        return predicted_labels

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
            loss = logistic_loss(logits.squeeze(1), labels)
            loss.backward()

            optimizer.step()
            train_losses.append(loss.item())

            dev_logits = model(dev_features)
            dev_loss = logistic_loss(dev_logits.squeeze(1), dev_labels)
            dev_losses.append(dev_loss.item())

        dev_f1 = f1_score(dev_labels.tolist(), predict(model, dev_features))

        print(f"epoch {i}")
        print(f"Train loss: {sum(train_losses) / len(train_losses)}")
        print(f"Dev loss: {sum(dev_losses) / len(dev_losses)}")
        print(f"Dev F1 {dev_f1}")

    return model, train_losses, dev_losses

train_features, train_labels_tensor = featurize_data(train_texts, train_labels)
train_features = standardize(train_features)
dev_features, dev_labels_tensor = featurize_data(dev_texts, dev_labels)
dev_features = standardize(dev_features)

num_features = 6
num_epochs = 30
model = SentimentClassifier(input_dim=num_features)
learning_rate = 0.01
optimizer = make_optimizer(model, learning_rate)

trained_model, train_losses, dev_losses = training_loop(
    num_epochs,
    15,
    train_features,
    train_labels_tensor,
    dev_features,
    dev_labels_tensor,
    optimizer,
    model
)

def write_final_predictions_csv(model, dev_texts, dev_labels, dev_ids, output_csv="subtask_1/pred_eng.csv"):
    features, _ = featurize_data(dev_texts, dev_labels)
    features = standardize(features)

    predicted_labels = predict(model, features)

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "polarization"])  # header
        for id_val, pred_label in zip(dev_ids, predicted_labels):
            writer.writerow([id_val, pred_label])

def generate_confusion_matrix():
    features, _ = featurize_data(dev_texts, dev_labels)
    features = standardize(features)

    predicted_labels = predict(model, features)

    cm = confusion_matrix(dev_labels.tolist(), predicted_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Subtask 1: German")
    plt.show()

#generate_confusion_matrix()

dev_datapath = "subtask1/dev/eng.csv"
dev_dataset = pd.read_csv(dev_datapath)
dev_texts = dev_dataset['text'].values
dev_labels = dev_dataset['polarization'].values
dev_ids = dev_dataset['id'].values

write_final_predictions_csv(trained_model, dev_texts, dev_labels, dev_ids)

