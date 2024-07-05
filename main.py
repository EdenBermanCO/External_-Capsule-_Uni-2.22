import numpy as np
import random
import json
import os

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Prepare data
def prepare_data(file):
    # Open the file and read its contents
    with open(file, "r") as f:
        data = f.readlines()

    # Initialize empty lists to store inputs and outputs
    inputs, outputs = [], []

    # Process each line in the file
    for line in data:
        # Split the line into input and output components
        split_line = line.split()

        # Convert the input string into a list of integers
        inputs.append([int(char) for char in split_line[0]])

        # Convert the output string into a single integer
        outputs.append([int(split_line[1])])

    # Convert the input and output lists into numpy arrays
    X = np.array(inputs)
    y = np.array(outputs)

    # Split the data into training and testing sets
    # using a test size of 20% and a fixed random seed of 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Return the train-test split data
    return X_train, X_test, y_train, y_test

class Agent:
    def __init__(self, network_architecture):
        self.network_architecture = network_architecture
        self.network = self.create_network()

    def create_network(self):
        network = []
        for i in range(len(self.network_architecture) - 1):
            network.append(2 * np.random.random((self.network_architecture[i], self.network_architecture[i + 1])) - 1)
        return network

    def mutate(self, mutation_rate):
        if random.random() < mutation_rate:
            self.mutate_architecture()
        else:
            self.mutate_weights(mutation_rate)

    def mutate_architecture(self):
        # Exclude input and output layers from mutation by selecting from hidden layers only
        if len(self.network_architecture) > 3:  # Ensure there are hidden layers to mutate

            # Select a random hidden layer index to mutate
            layer_idx_to_mutate = random.randint(1, len(self.network_architecture) - 2)

            # Randomly decide whether to remove a layer or add a layer
            if random.random() < 0.5:
                # Remove the selected layer
                del self.network_architecture[layer_idx_to_mutate]
                del self.network[layer_idx_to_mutate - 1]  # remove corresponding weights

                # Adjust weights for the next layer
                next_layer_weights = 2 * np.random.random((self.network_architecture[layer_idx_to_mutate - 1],
                                                           self.network_architecture[layer_idx_to_mutate])) - 1
                self.network[layer_idx_to_mutate - 1] = next_layer_weights
            else:
                # Add a new layer at the selected position
                new_node_count = random.randint(1, 10)
                self.network_architecture.insert(layer_idx_to_mutate, new_node_count)

                # Insert new random weights for the new layer
                new_layer_weights = 2 * np.random.random((self.network_architecture[layer_idx_to_mutate - 1],
                                                          new_node_count)) - 1
                self.network.insert(layer_idx_to_mutate - 1, new_layer_weights)

                # Adjust weights for the next layer
                next_layer_weights = 2 * np.random.random((new_node_count,
                                                           self.network_architecture[layer_idx_to_mutate + 1])) - 1
                self.network[layer_idx_to_mutate] = next_layer_weights
        else:
            # No hidden layers, add one
            layer_idx_to_mutate = 1
            new_node_count = random.randint(1, 10)
            self.network_architecture.insert(layer_idx_to_mutate, new_node_count)

            # Insert new random weights for the new layer
            new_layer_weights = 2 * np.random.random((self.network_architecture[layer_idx_to_mutate - 1],
                                                      new_node_count)) - 1
            self.network.insert(layer_idx_to_mutate - 1, new_layer_weights)

            # Adjust weights for the next layer
            next_layer_weights = 2 * np.random.random((new_node_count,
                                                       self.network_architecture[layer_idx_to_mutate + 1])) - 1
            self.network[layer_idx_to_mutate] = next_layer_weights

    def mutate_weights(self, mutation_rate):
        for layer in self.network:
            if layer.shape[1] > 1:
                if np.random.random() < mutation_rate:
                    index = np.random.randint(0, layer.shape[1] - 1)
                    # Modify the mutation to be smaller and proportional to the current weight
                    layer[:, index] += np.random.normal() * 0.05 * np.abs(layer[:, index])


class GeneticAlgorithm:
    def __init__(self, input_nodes, output_nodes, population, generations, threshold, mutation_rate):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.population = population
        self.generations = generations
        self.threshold = threshold
        self.mutation_rate = mutation_rate
        self.agents = []

    def create_population(self):
        self.agents = [Agent([self.input_nodes, random.randint(1, 10), self.output_nodes]) for _ in range(self.population)]

    def propagate(self, agent, data):
        for i in range(len(agent.network)):
            data = sigmoid(np.dot(data, agent.network[i]))
        return data

    # def fitness(self, X, y, X_test, y_test):
    #     for agent in self.agents:
    #         yhat = self.propagate(agent, X)
    #         cost = (yhat - y)**2
    #         agent.fitness = np.sum(cost)
    #
    #         # Also evaluate on test data for logging
    #         yhat_test = self.propagate(agent, X_test)
    #         cost_test = (yhat_test - y_test)**2
    #         agent.test_fitness = np.sum(cost_test)
    #
    #     self.agents = sorted(self.agents, key=lambda agent: agent.fitness, reverse=False)

    def fitness(self, X_train, y_train, X_test, y_test):
        epochs = 30
        for agent in self.agents:
            # Train the agent on the training set
            for _ in range(epochs):  # Specify the number of epochs for training
                for x, y in zip(X_train, y_train):
                    outputs = [x]  # Store the intermediate outputs of each layer
                    for i in range(len(agent.network)):
                        output = sigmoid(np.dot(outputs[-1], agent.network[i]))
                        outputs.append(output)

                    error = y - outputs[-1]
                    adjustments = [
                        error * sigmoid_derivative(outputs[-1])]  # Store the weight adjustments for each layer

                    # Backpropagation to calculate adjustments for each layer
                    for i in range(len(agent.network) - 2, -1, -1):
                        error = np.dot(adjustments[-1], agent.network[i + 1].T)
                        adjustment = error * sigmoid_derivative(outputs[i + 1])
                        adjustments.append(adjustment)

                    # Update the weights using the calculated adjustments
                    for i in range(len(agent.network)):
                        agent.network[i] += np.dot(outputs[i].reshape(-1, 1), adjustments[-(i + 1)].reshape(1, -1))

            # Evaluate the agent on the training set
            train_predictions = np.array([np.round(self.propagate(agent, x)) for x in X_train])
            train_accuracy = np.mean(train_predictions == y_train)
            agent.train_fitness = train_accuracy

            # Evaluate the agent on the test set
            test_predictions = np.array([np.round(self.propagate(agent, x)) for x in X_test])
            test_accuracy = np.mean(test_predictions == y_test)
            agent.test_fitness = test_accuracy

        self.agents = sorted(self.agents, key=lambda agent: agent.train_fitness, reverse=True)
        print("Generation Fitness:")
        for i, agent in enumerate(self.agents):
            print(f"Architecture {i + 1}: Train Accuracy: {agent.train_fitness}, Test Accuracy: {agent.test_fitness}")
    def crossover(self):
        offspring = []
        for _ in range(len(self.agents) // 2):
            parent1 = random.choice(self.agents[:10])
            parent2 = random.choice(self.agents[:10])

            # Perform a one-point crossover on the network architecture
            crossover_point = random.randint(1, min(len(parent1.network_architecture), len(parent2.network_architecture)) - 1)
            child_architecture = parent1.network_architecture[:crossover_point] + parent2.network_architecture[crossover_point:]
            child = Agent(child_architecture)
            offspring.append(child)

        self.agents = self.agents[:self.population // 2] + offspring

    def mutation(self):
        for agent in self.agents:
            agent.mutate(self.mutation_rate)

    # def execute(self, X_train, y_train, X_test, y_test):
    #     self.create_population()
    #     for generation in range(self.generations):
    #         print(f"Generation {generation+1}/{self.generations}")
    #         self.fitness(X_train, y_train, X_test, y_test)
    #         best_agent = self.agents[0]
    #         print(f"Best Agent's train fitness: {best_agent.fitness}, test fitness: {best_agent.test_fitness}")
    #         if best_agent.fitness < self.threshold:
    #             print(f"Threshold met at Generation {generation+1}")
    #             return best_agent
    #         self.crossover()
    #         self.mutation()
    #     return best_agent
    def execute(self, X_train, y_train, X_test, y_test):
        self.create_population()
        for generation in range(self.generations):
            print(f"Generation {generation + 1}/{self.generations}")
            self.fitness(X_train, y_train, X_test, y_test)
            best_agent = self.agents[0]
            print(f"Best Agent's train accuracy: {best_agent.train_fitness}, test accuracy: {best_agent.test_fitness}")
            if best_agent.train_fitness >= 0.99:  # Modify the condition based on your desired accuracy threshold
                print(f"Desired accuracy reached at Generation {generation + 1}")
                return best_agent
            self.crossover()
            self.mutation()
        return best_agent


def save_network(agent, filename):
    network_dict = {"architecture": agent.network_architecture, "weights": [layer.tolist() for layer in agent.network]}
    with open(filename, "w") as f:
        json.dump(network_dict, f)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data("nn0.txt")
    input_nodes = X_train.shape[1]
    output_nodes = y_train.shape[1]
    ga = GeneticAlgorithm(input_nodes, output_nodes, population=20, generations=100, threshold=0.5, mutation_rate=0.1)
    best_agent = ga.execute(X_train, y_train, X_test, y_test)
    save_network(best_agent, "wnet.txt")
    print(f"Best Agent's fitness: {best_agent.fitness}, test fitness: {best_agent.test_fitness}")
