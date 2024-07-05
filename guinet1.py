import tkinter as tk
import numpy as np
import buildnet0
import threading
import queue
import sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import messagebox

message_queue = []  # Store the messages
current_message_index = 0  # Index of the current message being displayed
# Set default values for the global variables
default_train_file = 'train_nn1.txt'
default_test_file = 'test_nn1.txt'


def prepare_data(train_file, test_file):
    with open(train_file, "r") as f:
        train_data = f.readlines()

    with open(test_file, "r") as f:
        test_data = f.readlines()

    train_inputs, train_outputs = [], []
    for line in train_data:
        split_line = line.split()
        train_inputs.append([int(char) for char in split_line[0]])
        train_outputs.append([int(split_line[1])])

    test_inputs, test_outputs = [], []
    for line in test_data:
        split_line = line.split()
        test_inputs.append([int(char) for char in split_line[0]])
        test_outputs.append([int(split_line[1])])

    X_train = np.array(train_inputs)
    y_train = np.array(train_outputs)
    X_test = np.array(test_inputs)
    y_test = np.array(test_outputs)

    return X_train, X_test, y_train, y_test


def check_queue(q):
    global current_message_index

    try:
        msg = q.get_nowait()  # try to get a message from the queue
    except queue.Empty:
        pass  # if there's nothing in the queue, do nothing
    else:
        if msg[0] == 'result':
            # If the message is a 'result' message, show the messagebox with the results
            fitness_calls, number_generations, final_fitness_score, final_solutions, accuracyTest, accuracyTrain = msg[1], msg[2], msg[3], msg[4], msg[5], msg[6]

            messagebox.showinfo("Results", f"Loss(fittness):  {fitness_calls:.3f}\n"
                                           f"Precision:  {number_generations:.3f}\n"
                                           f"Recall:  {final_fitness_score:.3f}\n"                           
                                           f"F-score:  {final_solutions:.3f}\n"
                                           f"Accuracy (Test Set):  {accuracyTest:.3f}\n"
                                           f"Accuracy (Train Set):  {accuracyTrain:.3f}")
        else:
            # Otherwise, add the message to the queue as before
            message_queue.append(msg)

    if current_message_index < len(message_queue):
        output_text.insert(tk.END, message_queue[current_message_index] + '\n')  # display the current message
        output_text.see(tk.END)  # scroll to the last line
        current_message_index += 1

    if current_message_index >= len(message_queue):
        root.after(100, check_queue, q)  # wait for 5 seconds before checking the queue again
    else:
        root.after(100, check_queue, q)  # check the queue again after 1 ms

def on_close():
    print("Closing program... Thank you for checking the assigment!")
    root.destroy()
    sys.exit()

def run_genetic_algorithm():
    global current_message_index
    message_queue.clear()
    output_text.delete(1.0, tk.END)
    current_message_index = 0  # Reset the current message index
    q = queue.Queue()
    X_train, X_test, y_train, y_test = prepare_data(default_train_file, default_test_file)

    input_dim = X_train.shape[1]
    hidden1_dim = 8
    hidden2_dim = 5
    hidden3_dim = 3
    output_dim = y_train.shape[1]
    population_size = 200
    generations = 250

    # Start the execution in a new thread
    thread = threading.Thread(target=buildnet0.execute, args=(
        q, fig_graphs, ax1, fig_tsne_before, ax_tsne_before, fig_tsne_after, ax_tsne_after, canvas1, canvas2, canvas3,
        X_train, X_test, y_train, y_test, input_dim,
        hidden1_dim, hidden2_dim, hidden3_dim, output_dim, population_size, generations))
    thread.daemon = True
    thread.start()
    check_queue(q)  # check the queue for messages right away


root = tk.Tk()

# Create a frame to hold the input fields
input_frame = tk.Frame(root)
input_frame.pack()

# Create a frame to hold the figures
figures_frame = tk.Frame(root)
figures_frame.pack()

# Create the figure and axis for live update graphs
fig_graphs = Figure(figsize=(5, 5), dpi=100)
ax1 = fig_graphs.add_subplot(111)
canvas1 = FigureCanvasTkAgg(fig_graphs, master=figures_frame)
canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Create the figure for t-SNE before training
fig_tsne_before = Figure(figsize=(5, 5), dpi=100)
ax_tsne_before = fig_tsne_before.add_subplot(111)
canvas2 = FigureCanvasTkAgg(fig_tsne_before, master=figures_frame)
canvas2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Create the figure for t-SNE after training
fig_tsne_after = Figure(figsize=(5, 5), dpi=100)
ax_tsne_after = fig_tsne_after.add_subplot(111)
canvas3 = FigureCanvasTkAgg(fig_tsne_after, master=figures_frame)
canvas3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

run_button = tk.Button(root, text="Run", command=run_genetic_algorithm)
run_button.pack()

# We choose the width by the amount of characters.
output_text = tk.Text(root, width=130)
output_text.pack()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
