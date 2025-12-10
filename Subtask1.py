import torch
import pandas as pd
import math

from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import word_count
from tf_idf import pol_top_tfidf_eng, nonpol_top_tfidf_eng
import keyness

datapath = "subtask1/train/eng.csv"

dataset = pd.read_csv(datapath)

#load data into separate pandas dataframes based on corresponding conditions
pol_data = dataset[dataset['polarization'] == 1]
nonpol_data = dataset[dataset['polarization'] == 0]

all_texts = dataset['text'].values
all_labels = dataset['polarization'].values

def split_dataset(texts, labels):
    tr_texts, de_texts, tr_labels, de_labels = train_test_split(texts, labels, test_size=.2, random_state=42)
    return tr_texts, de_texts, tr_labels, de_labels

train_texts, dev_texts, train_labels, dev_labels = split_dataset(all_texts, all_labels)

def preprocess(text):
    """
    Takes a string of text and returns a list of the whitespace-separated and lowercased tokens.
    """

    tokens = text.lower().split()

    return tokens

# Read in positive and negative words from the text files
path_to_negatives = "negative-words.txt"
negative_words = []
high_freq_pol_words = word_count.pol_words
high_freq_nonpol_words = word_count.nonpol_words
pol_words = pol_top_tfidf_eng
nonpol_words = nonpol_top_tfidf_eng

with open(path_to_negatives, "r") as file:
    for line in file:
        text = line.rstrip()
        negative_words.append(text)

def get_negative_tokens(text):
    negative_tokens_found = [token for token in text if token in negative_words]
    return len(negative_tokens_found)

def get_polarizing_tokens(text):
    polarizing_tokens_found = [token for token in text if token in pol_words]
    return len(polarizing_tokens_found)

def get_nonpolarizing_tokens(text):
    nonpolarizing_tokens_found = [token for token in text if token in nonpol_words]
    return len(nonpolarizing_tokens_found)

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
    features.append(get_polarizing_tokens(text))
    features.append(get_nonpolarizing_tokens(text))
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
    """
    Return the features transformed by the above standardization formula
    """
    num_columns = features.shape[1]
    means = []
    stds = []
    scaled_features = []

    # get means
    for col in range(num_columns):
        column = features[:, col]
        total = 0
        for row in range(len(column)):
            total += features[row][col]
        means.append(total / len(column))

    # get stds
    for col in range(num_columns):
        column = features[:, col]
        total = 0
        for row in range(len(column)):
            total += pow((features[row][col] - means[col]), 2)
        stds.append(math.sqrt(total / len(column)))

    # standardize
    for col in range(num_columns):
        column = features[:, col]
        for row in range(len(column)):
            features[row][col] = (column[row] - means[col]) / stds[col]

    return features


class SentimentClassifier(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # We force output to be one, since we are doing binary logistic regression
        self.output_size = 1
        self.coefficients = torch.nn.Linear(input_dim, self.output_size)
        # Initialize weights. Note that this is not strictly necessary,
        # but you should test different initializations per lecture
        initialize_weights(self.coefficients)

    def forward(self, features: torch.Tensor):
        # We predict a number by multiplying by the coefficients
        # and then take the sigmoid to turn the score as logits
        return torch.sigmoid(self.coefficients(features))


def initialize_weights(coefficients):
    """
    Initialize the weights of the coefficients to ones.
    """
    torch.nn.init.ones_(coefficients.weight)

    return coefficients


def logistic_loss(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Logistic loss function between a prediction and label.
    """
    loss = torch.nn.BCELoss()
    output = loss(prediction, label)

    return output

def make_optimizer(model, learning_rate) -> torch.optim:
    """
    Returns a Stochastic Gradient Descent Optimizer
    See here for algorithms you can import: https://pytorch.org/docs/stable/optim.html
    """
    return torch.optim.SGD(model.parameters(), learning_rate)


def predict(model, features):
    with torch.no_grad():
        """
        Implement the logic of converting the logits into prediction labels (0, 1). Return as a Python list.
        """
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
            # Empty the dynamic computation graph
            features, labels = zip(*batch)
            features = torch.stack(features)
            labels = torch.stack(labels)
            optimizer.zero_grad()
            # Run the model
            logits = model(features)
            # Compute loss
            loss = logistic_loss(torch.squeeze(logits), labels)
            # In this logistic regression example,
            # this entails computing a single gradient
            loss.backward()
            # Backpropogate the loss through our model

            # Update our coefficients in the direction of the gradient.
            optimizer.step()
            # For logging
            train_losses.append(loss.item())

            # Compute dev loss for our reference
            dev_logits = model(dev_features)
            dev_loss = logistic_loss(torch.squeeze(dev_logits), dev_labels)
            dev_losses.append(dev_loss.item())

        # Estimate the f1 score for the development set
        dev_f1 = f1_score(dev_labels.tolist(), predict(model, dev_features))
        print(f"epoch {i}")
        print(f"Train loss: {sum(train_losses) / len(train_losses)}")
        print(f"Dev loss: {sum(dev_losses) / len(dev_losses)}")
        print(f"Dev F1 {dev_f1}")

    # Return the trained model
    return model, train_losses, dev_losses


# run the model
train_features, train_labels_tensor = featurize_data(train_texts, train_labels)
train_features = standardize(train_features)
dev_features, dev_labels_tensor = featurize_data(dev_texts, dev_labels)
dev_features = standardize(dev_features)

# Initialize the model, optimizer and num_epochs here
# Epochs and learning rate should be experimented with
# The model is the SentimentClassifier

num_features = len(extract_features(train_texts))
num_epochs = 30
model = SentimentClassifier(input_dim=num_features)
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