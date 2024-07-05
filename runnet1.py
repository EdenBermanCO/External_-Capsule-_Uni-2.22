import numpy as np
import os
from tkinter import Tk, Label, Button, Entry, StringVar, DISABLED, W
import tkinter.messagebox as messagebox

from sklearn.metrics import precision_recall_fscore_support


class Gui:
    def __init__(self, master):
        self.master = master
        master.title("RUNNET1 - Neural Network Prediction")

        self.label1 = Label(master, text="True Labels File (optional to evaluate the model):")
        self.label2 = Label(master, text="Predictions File (output):")
        self.label3 = Label(master, text="Test Data File (input):")
        self.label4 = Label(master, text="Model Name:")

        self.true_labels_file = StringVar(value="")
        self.predictions_file = StringVar(value="nn1_prediction.txt")
        self.test_data_file = StringVar(value="testnet1.txt")
        self.model_name = StringVar(value="wnet1.npz")

        self.entry1 = Entry(master, textvariable=self.true_labels_file)
        self.entry2 = Entry(master, textvariable=self.predictions_file)
        self.entry3 = Entry(master, textvariable=self.test_data_file)
        self.entry4 = Entry(master, textvariable=self.model_name)

        self.entry4.config(state=DISABLED)  # Make the model name field read-only

        self.label1.grid(row=0, column=0, sticky=W)
        self.label2.grid(row=1, column=0, sticky=W)
        self.label3.grid(row=2, column=0, sticky=W)
        self.label4.grid(row=3, column=0, sticky=W)

        self.entry1.grid(row=0, column=1)
        self.entry2.grid(row=1, column=1)
        self.entry3.grid(row=2, column=1)
        self.entry4.grid(row=3, column=1)

        self.run_button = Button(master, text="Run", command=self.run)
        self.run_button.grid(row=4, column=0, columnspan=2)

    def run(self):
        # Call your main function here
        network_file = self.model_name.get()
        data_file = self.test_data_file.get()
        predictions_file = self.predictions_file.get()
        true_labels_file = self.true_labels_file.get()

        runnet(network_file, data_file, predictions_file)

        # if the True Labels File exists and is not empty
        if os.path.exists(true_labels_file) and os.path.getsize(true_labels_file) > 0:
            correct, incorrect, accuracy, true_labels, predictions = compare_files(true_labels_file, predictions_file)

            precision, recall, fscore, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')

            result = f"Correct predictions: {correct}\n" \
                     f"Incorrect predictions: {incorrect}\n" \
                     f"Accuracy: {accuracy:.2f}%\n" \
                     f"Precision: {precision:.2f}\n" \
                     f"Recall: {recall:.2f}\n" \
                     f"F-score: {fscore:.2f}"
            messagebox.showinfo("Result", result)
        else:  # if the True Labels File is empty
            messagebox.showinfo("Result", "Finished calculating the predictions. "
                                          "Unfortunately, you didn't provide the true labels for the samples, "
                                          "so we can't evaluate the model.")


class Agent:
    def __init__(self, network):
        self.neural_network = network
        self.fitness = 0

class NeuralNetwork:
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, dropout_rate=0.2):
        self.weights = [
            np.random.randn(hidden1_dim, input_dim) / np.sqrt(input_dim),
            np.random.randn(hidden2_dim, hidden1_dim) / np.sqrt(hidden1_dim),
            np.random.randn(hidden3_dim, hidden2_dim) / np.sqrt(hidden2_dim),
            np.random.randn(output_dim, hidden3_dim) / np.sqrt(hidden3_dim),
        ]

        self.biases = [
            np.random.randn(hidden1_dim, 1),
            np.random.randn(hidden2_dim, 1),
            np.random.randn(hidden3_dim, 1),
            np.random.randn(output_dim, 1)
        ]
        self.dropout_rate = dropout_rate
        # New instance variables for batch norm
        self.bn_means = [np.zeros((1, dim)) for dim in [hidden1_dim, hidden2_dim, hidden3_dim, output_dim]]
        self.bn_vars = [np.zeros((1, dim)) for dim in [hidden1_dim, hidden2_dim, hidden3_dim, output_dim]]
        self.bn_decay = 0.9  # Decay rate for the running averages

    def propagate(self, X, training=True, return_hidden=False):
        X = np.array(X).reshape(-1, self.weights[0].shape[1])  # Ensure X has the correct shape
        hidden1 = self.elu(np.dot(X, self.weights[0].T) + self.biases[0].T)

        hidden2 = self.relu(np.dot(hidden1, self.weights[1].T) + self.biases[1].T)
        hidden2 = self.batch_norm(hidden2, 1, training)  # Specify layer index and training

        hidden3 = self.elu(np.dot(hidden2, self.weights[2].T) + self.biases[2].T)
        hidden3 = self.dropout(hidden3, self.dropout_rate, training)

        output_layer = self.sigmoid(np.dot(hidden3, self.weights[3].T) + self.biases[3].T)

        # If we're interested in the activations of the last hidden layer for draw TSNE:
        if return_hidden:
            return hidden3
        else:
            return output_layer

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # The alpha parameter controls the value that ELU converges towards for negative net inputs
    def elu(self, x, alpha=1.0):
        return np.where(x >= 0.0, x, alpha * (np.exp(x) - 1))

    def dropout(self, X, dropout_rate, training=True):
        if not training:
            return X
        keep_prob = 1 - dropout_rate
        mask = np.random.binomial(1, keep_prob, size=X.shape) / keep_prob
        return X * mask

    def batch_norm(self, X, layer, training=True):
        if training:
            mean = np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)

            # Update running averages
            self.bn_means[layer] = self.bn_decay * self.bn_means[layer] + (1 - self.bn_decay) * mean
            self.bn_vars[layer] = self.bn_decay * self.bn_vars[layer] + (1 - self.bn_decay) * var
        else:
            # Use running averages
            mean = self.bn_means[layer]
            var = self.bn_vars[layer]

        X_norm = (X - mean) / np.sqrt(var + 1e-8)
        return X_norm


def load_network(filename):
    # Load the dictionary from the numpy .npz file
    network_dict = np.load(filename)

    # Extract the architecture information
    input_dim = network_dict['input_dim'].item()
    hidden1_dim = network_dict['hidden1_dim'].item()
    hidden2_dim = network_dict['hidden2_dim'].item()
    hidden3_dim = network_dict['hidden3_dim'].item()
    output_dim = network_dict['output_dim'].item()
    dropout_rate = network_dict['dropout_rate'].item()

    # Create a new network with the loaded architecture
    network = NeuralNetwork(input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, dropout_rate)

    # Extract the weights and biases
    weights = [network_dict[f'weight_{i}'] for i in range(len(network.weights))]
    biases = [network_dict[f'bias_{i}'] for i in range(len(network.biases))]
    bn_means = [network_dict[f'bn_mean_{i}'] for i in range(len(network.bn_means))]
    bn_vars = [network_dict[f'bn_var_{i}'] for i in range(len(network.bn_vars))]

    # Assign the loaded weights and biases to the network
    network.weights = weights
    network.biases = biases
    network.bn_means = bn_means
    network.bn_vars = bn_vars

    return network


def predict(network, X):
    predictions = network.propagate(X, training=False)
    predicted_labels = np.round(predictions)
    return predicted_labels

def save_predictions(predictions, filename):
    np.savetxt(filename, predictions, fmt='%d')

    # Open the file in binary mode, go to the end
    with open(filename, 'rb+') as file:
        file.seek(-2, os.SEEK_END)
        file.truncate()  # Truncate file at current position


def runnet(network_file, data_file, output_file):
    # Load the network from the file
    network = load_network(network_file)

    # Load the unlabeled data
    with open(data_file, "r") as f:
        data = f.readlines()

    X = []
    for line in data:
        split_line = line.split()
        X.append([int(char) for char in split_line[0]])

    X = np.array(X)
    # Predict the labels of the unlabeled data
    predictions = predict(network, X)
    # Save the predicted labels to a file
    save_predictions(predictions, output_file)


def compare_files(true_labels_file, predictions_file):
    with open(true_labels_file, 'r') as file1, open(predictions_file, 'r') as file2:
        true_labels = [line.strip() for line in file1.readlines()]
        predictions = [line.strip() for line in file2.readlines()]

    correct = 0
    incorrect = 0

    for true_label, prediction in zip(true_labels, predictions):
        if true_label == prediction:
            correct += 1
        else:
            incorrect += 1

    accuracy = correct / (correct + incorrect) * 100  # Calculating accuracy
    # convert to int to use them to calcualte Precision, Recall and F-score.
    true_labels = [int(label) for label in true_labels]
    predictions = [int(label) for label in predictions]
    return correct, incorrect, accuracy, true_labels, predictions



if __name__ == "__main__":
    root = Tk()
    gui = Gui(root)
    root.mainloop()