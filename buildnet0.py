import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.manifold import TSNE

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

    def elu(self, x,
            alpha=1.0):  # The alpha parameter controls the value that ELU converges towards for negative net inputs
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


def generate_agents(population, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim):
    return [Agent(NeuralNetwork(input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim)) for _ in range(population)]


def selection(agents):
    agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False)
    agents = agents[:int(0.2 * len(agents))]
    return agents


def unflatten(flattened, shapes):
    newarray = []
    index = 0
    for shape in shapes:
        size = np.product(shape)
        newarray.append(flattened[index : index + size].reshape(shape))
        index += size
    return newarray


def blend_crossover(alpha, parent1, parent2):
    # Create copies of the parents' genes
    genes1 = parent1.flatten()
    genes2 = parent2.flatten()

    # Initialize children genes with parent genes
    child1_genes = genes1.copy()
    child2_genes = genes2.copy()

    # Apply blend crossover to each gene
    for i in range(len(genes1)):
        # Calculate lower and upper bounds for the new genes
        lower = min(genes1[i], genes2[i]) - alpha * abs(genes1[i] - genes2[i])
        upper = max(genes1[i], genes2[i]) + alpha * abs(genes1[i] - genes2[i])

        # Generate the new genes by picking a random value between the lower and upper bounds
        child1_genes[i] = np.random.uniform(lower, upper)
        child2_genes[i] = np.random.uniform(lower, upper)

    # Reshape child genes to parent gene shapes
    child1_genes = child1_genes.reshape(parent1.shape)
    child2_genes = child2_genes.reshape(parent2.shape)

    return child1_genes, child2_genes


def crossover(input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, agents, pop_size, alpha=0.5, best_agent=None):
    offspring = []
    num_offspring = pop_size - len(agents)
    for _ in range(num_offspring // 2):
        parent1 = random.choice(agents)
        parent2 = random.choice(agents)

        child1 = Agent(NeuralNetwork(input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim))
        child2 = Agent(NeuralNetwork(input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim))

        child1_weights = []
        child2_weights = []
        for w1, w2 in zip(parent1.neural_network.weights, parent2.neural_network.weights):
            child1_w, child2_w = blend_crossover(alpha, w1, w2)
            child1_weights.append(child1_w)
            child2_weights.append(child2_w)

        child1_biases = []
        child2_biases = []
        for b1, b2 in zip(parent1.neural_network.biases, parent2.neural_network.biases):
            child1_b, child2_b = blend_crossover(alpha, b1, b2)
            child1_biases.append(child1_b)
            child2_biases.append(child2_b)

        child1.neural_network.weights = child1_weights
        child2.neural_network.weights = child2_weights
        child1.neural_network.biases = child1_biases
        child2.neural_network.biases = child2_biases

        offspring.append(child1)
        offspring.append(child2)

    agents = agents[:pop_size - len(offspring)] + offspring

    # Elitism: If there is a best agent, replace the worst performing agent
    if best_agent is not None:
        agents.sort(key=lambda agent: agent.fitness, reverse=True)
        agents[0] = best_agent

    return agents


def mutation(agents):
    for agent in agents:
        if random.uniform(0.0, 1.0) <= 0.5:
            weights = agent.neural_network.weights
            biases = agent.neural_network.biases
            shapes = [a.shape for a in weights] + [b.shape for b in biases]
            flattened = np.concatenate([a.flatten() for a in weights] + [b.flatten() for b in biases])
            # random index will be used to select a random element for mutation.
            randint = random.randint(0, len(flattened) - 1)
            # The value at the randomly selected index in flattened is replaced
            # with a new random value generated using np.random.randn().
            flattened[randint] = np.random.randn()
            newarray = []
            indeweights = 0
            for shape in shapes:
                size = np.product(shape)
                newarray.append(flattened[indeweights : indeweights + size].reshape(shape))
                indeweights += size
            agent.neural_network.weights = newarray[:len(weights)]
            agent.neural_network.biases = newarray[len(weights):]
    return agents

def calculate_accuracy(agent, X, y, isTraining = True):
    predictions = agent.neural_network.propagate(X, isTraining)
    predicted_labels = np.round(predictions)
    accuracy = np.mean(predicted_labels == y)
    return accuracy


def fitness(agents, X, y, batch_size):
    epsilon = 1e-7  # To prevent division by zero
    num_samples = X.shape[0]
    for agent in agents:
        log_loss_list = []
        for i in range(0, num_samples, batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            yhat = agent.neural_network.propagate(X_batch)
            yhat = np.clip(yhat, epsilon, 1. - epsilon)  # Ensure yhat is within [epsilon, 1-epsilon]
            log_loss = -np.mean(y_batch * np.log(yhat) + (1 - y_batch) * np.log(1 - yhat))
            log_loss_list.append(log_loss)

        agent.fitness = np.mean(log_loss_list)  # Average log loss over all batches
    return agents

def visualize(q, fig, ax, canvas, agent, h, color, title, training=True):
    h_transformed = agent.neural_network.propagate(h, training, return_hidden=True)
    z = TSNE(n_components=2).fit_transform(h_transformed)
    scatter = ax.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.add_artist(legend1)
    ax.set_title(title)
    canvas.draw_idle()
    plt.pause(0.5)

    if(training):
        # Save the figure
        fig.savefig('TSNE_Before_nn0.png')
    else:
        fig.savefig('TSNE_After_nn0.png')

def execute(q, fig_graphs, ax1, fig_tsne_before, ax_tsne_before, fig_tsne_after, ax_tsne_after, canvas1, canvas2, canvas3, X_train, X_test, y_train, y_test, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, population_size, generations):
    agents = generate_agents(population_size, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim)
    batch_size = 512
    best_solution = agents[0]
    # Clear the axes for the new plot
    ax1.clear()
    q.put(f'Please wait while we render the t-SNE results...')

    # We will keep track of loss and accuracy for each generation in these lists
    losses = []
    accuracies = []
    iterations = []
    train_loss = 0
    train_accuracy = 0
    # Visualize the data model before training
    visualize(q, fig_tsne_before, ax_tsne_before, canvas2, best_solution, X_train, y_train, "Data Model - Before Training", training=True)
    canvas2.draw()
    gen = 0
    agents = fitness(agents, X_train, y_train, batch_size)
    for i in range(generations):
        q.put(f'Generation {i} :')

        # Store values to build the graph over each iteration.
        losses.append(train_loss)
        accuracies.append(train_accuracy)
        iterations.append(i)

        ax1.clear()   # clear the plot for the new plot
        ax1.plot(iterations, losses, 'r-', label='Loss')  # plot Loss with red line
        ax1.plot(iterations, accuracies, 'b-', label='Accuracy')  # plot Accuracy score with blue line
        ax1.set_xlabel('Generations')  # Set the x-axis label
        ax1.set_ylabel('Evaluate Training Phase')  # Set the y-axis label
        ax1.legend()

        # Ask the canvas to redraw itself the next time it can
        canvas1.draw()  # refresh the canvas

        agents = selection(agents)
        # Apply crossover and mutation
        agents = crossover(input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, agents, population_size, best_agent=best_solution)
        agents = mutation(agents)
        agents = fitness(agents, X_train, y_train, batch_size)

        best_agent = min(agents, key=lambda agent: agent.fitness)
        if best_agent.fitness < best_solution.fitness:
            best_solution = best_agent

        train_loss = best_agent.fitness
        train_accuracy = calculate_accuracy(best_agent, X_train, y_train)

        q.put(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")
        gen = i
        if train_accuracy > 0.992 and train_loss < 0.02:
            break

    bestTestScore = -np.inf
    bestLoss = np.inf
    bestNeuralNetwork = best_solution
    for agent in agents:
        isTrain = False
        train_loss = agent.fitness
        test_accuracy = calculate_accuracy(agent, X_test, y_test, isTrain)
        if test_accuracy > bestTestScore:
            bestTestScore = test_accuracy
            bestNeuralNetwork = agent
        if test_accuracy == bestTestScore and bestLoss > train_loss:
            bestNeuralNetwork = agent
        train_accuracy = calculate_accuracy(agent, X_train, y_train, isTrain)

        q.put(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")


    isTrain = False
    train_loss = bestNeuralNetwork.fitness
    test_accuracy = calculate_accuracy(bestNeuralNetwork, X_test, y_test, isTrain)
    train_accuracy = calculate_accuracy(bestNeuralNetwork, X_train, y_train, isTrain)
    q.put(f"Chosen model values (after {gen} generations): ")
    q.put(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    y_pred = bestNeuralNetwork.neural_network.propagate(X_test)
    y_pred = np.round(y_pred)  # convert probabilities to class labels
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    q.put(f'Please wait while we render the t-SNE results and evaluate the model...')
    visualize(q, fig_tsne_after, ax_tsne_after, canvas3, bestNeuralNetwork, X_train, y_train, "Data Model - After Training", training=False)
    canvas3.draw()
    q.put(f'Done!')
    # Add the new metrics to the GUI
    q.put(('result', train_loss, precision, recall, fscore, test_accuracy, train_accuracy))
    fig_graphs.savefig('Graph_Results_nn0.png')
    save_network(bestNeuralNetwork, "wnet")
    return bestNeuralNetwork

def save_network(agent, filename):
    # Save weights and biases into dictionary
    network_dict = {}
    for i, (weight, bias) in enumerate(zip(agent.neural_network.weights, agent.neural_network.biases)):
        network_dict[f'weight_{i}'] = weight
        network_dict[f'bias_{i}'] = bias
        network_dict[f'bn_mean_{i}'] = agent.neural_network.bn_means[i]  # Save batch normalization mean
        network_dict[f'bn_var_{i}'] = agent.neural_network.bn_vars[i]  # Save batch normalization variance

    # Save architecture information
    network_dict['input_dim'] = agent.neural_network.weights[0].shape[1]
    network_dict['hidden1_dim'] = agent.neural_network.weights[0].shape[0]
    network_dict['hidden2_dim'] = agent.neural_network.weights[1].shape[0]
    network_dict['hidden3_dim'] = agent.neural_network.weights[2].shape[0]
    network_dict['output_dim'] = agent.neural_network.weights[3].shape[0]
    network_dict['dropout_rate'] = agent.neural_network.dropout_rate

    # Save the dictionary to a numpy .npz file
    np.savez(filename, **network_dict)
