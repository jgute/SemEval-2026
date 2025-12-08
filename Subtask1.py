import numpy as np
import random
import torch
from typing import List
import pandas as pd
import math
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import word_count
from tf_idf import pol_top_tfidf
from tf_idf import nonpol_top_tfidf
from word2vec import w2v_model

datapath = "subtask1/train/eng.csv"

dataset = pd.read_csv(datapath)

#load data into separate pandas dataframes based on corresponding conditions
pol_data = dataset[dataset['polarization'] == 1]
nonpol_data = dataset[dataset['polarization'] == 0]

all_texts = dataset['text'].values
all_labels = dataset['polarization'].values

#validate data load
def validate_data():
    print("example data:")
    for i in range(3):
        print(pol_data['text'].values[i], "(polarization = 1)")
        print(nonpol_data['text'].values[i], "(polarization = 0)")

    print(f"There are {len(pol_data)} positive labels.")
    print(f"There are {len(nonpol_data)} negative labels.")
    print(all_labels)

#validate_data()

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
#pol_words = word_count.pol_words
#nonpol_words = word_count.nonpol_words
pol_words = pol_top_tfidf
nonpol_words = nonpol_top_tfidf
# print(pol_words)
with open(path_to_negatives, "r") as file:
    for line in file:
        text = line.rstrip()
        negative_words.append(text)

def get_negative_tokens(text):
    negative_tokens_found = [token for token in text if token in negative_words]
    return len(negative_tokens_found)

def get_w2f_vectors(text):
    vectors = []
    for word in text:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])
    return vectors

def get_avg_vector(text):
    vecs = get_w2f_vectors(text)
    if len(vecs) > 0:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)

def get_polarizing_tokens(text):
    polarizing_tokens_found = [token for token in text if token in pol_words]
    return len(polarizing_tokens_found)

def get_word_count(text):
    return len(text)

def get_nonpolarizing_tokens(text):
    nonpolarizing_tokens_found = [token for token in text if token in nonpol_words]
    return len(nonpolarizing_tokens_found)

def extract_features(text):
    features = []
    features.append(get_negative_tokens(text))
    features.append(get_polarizing_tokens(text))
    features.append(get_nonpolarizing_tokens(text))
    #features.append(get_avg_vector(text))
    features.append(get_word_count(text))

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
        # We predict a number by multipling by the coefficients
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


def accuracy(predicted_labels, true_labels):
    """
    Accuracy is correct predictions / all predicitons

    Args:
        predicted_labels (np.ndarray[int, 1]): the integer labels from the predictions. Uni-dimensional
        true_labels (np.ndarray[int, 1]): the integer labels from the gold standard. Uni-dimensional

    Returns:
        accuracy_value (double)

    """
    accuracy_value = 0.
    correct_predictions = 0
    all_predictions = len(predicted_labels)

    for i in range(len(predicted_labels)):
        if predicted_labels[i] == true_labels[i]:
            correct_predictions += 1

    accuracy_value = correct_predictions / all_predictions

    return accuracy_value


def precision(predicted_labels, true_labels):
    """
    Precision is True Positives / All Positives Predictions

    Args:
        predicted_labels (np.ndarray[int, 1]): the integer labels from the predictions. Uni-dimensional
        true_labels (np.ndarray[int, 1]): the integer labels from the gold standard. Uni-dimensional

    Returns:
        precision_value (double)

    """
    precision_value = 0.
    true_positives = 0
    all_positives = 0

    for i in range(len(predicted_labels)):
        if predicted_labels[i] == 1:
            all_positives += 1
        if predicted_labels[i] == true_labels[i] and predicted_labels[i] == 1:
            true_positives += 1

    precision_value = true_positives / all_positives

    return precision_value


def recall(predicted_labels, true_labels):
    """
    Recall is True Positives / All Positive Labels

    Args:
        predicted_labels (np.ndarray[int, 1]): the integer labels from the predictions. Uni-dimensional
        true_labels (np.ndarray[int, 1]): the integer labels from the gold standard. Uni-dimensional

    Returns:
        recall_value (double)

    """
    recall_value = 0.
    true_positives = 0
    all_positives = 0

    for i in range(len(predicted_labels)):
        if true_labels[i] == 1:
            all_positives += 1
        if predicted_labels[i] == true_labels[i] and predicted_labels[i] == 1:
            true_positives += 1

    recall_value = true_positives / all_positives

    return recall_value


def f1_score(predicted_labels, true_labels):
    """
    F1 score is the harmonic mean of precision and recall

    Args:
        predicted_labels (np.ndarray[int, 1]): the integer labels from the predictions. Uni-dimensional
        true_labels (np.ndarray[int, 1]): the integer labels from the gold standard. Uni-dimensional

    Returns:
        f1_score_value (double)

    """
    f1_score_value = 0.

    denominator = (1 / precision(predicted_labels, true_labels)) + (1 / recall(predicted_labels, true_labels))
    f1_score_value = 2 / denominator

    return f1_score_value

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
        dev_f1 = f1_score(predict(model, dev_features), dev_labels.tolist())
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
num_epochs = 100
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

# Random Classifier
def predict_random(train_labels, num_samples):
   """
   Using the label distribution, predict the label num_sample number of times


   Args:
       train_labels np.ndarray(int)
       num_samples: int
   Returns:
       predictions np.ndarray(int, num_samples)
   """


   pos_count = 0
   neg_count = 0


   for label in train_labels:
       if label == 1:
           pos_count += 1
       else:
           neg_count += 1

   pos_prob = pos_count / (pos_count + neg_count)

   predictions = random.choices(population=[1, 0], weights=[pos_prob, (1 - pos_prob)], k=num_samples)

   return np.array(predictions, dtype=int)

# random classifier baseline
devset_prediction_random = predict_random(train_labels, num_samples=len(dev_labels))
dev_random_f1 = f1_score(devset_prediction_random, dev_labels)
print('\nRandom Chance F1:', dev_random_f1)
# should be 0.5, but is instead ~0.3???
