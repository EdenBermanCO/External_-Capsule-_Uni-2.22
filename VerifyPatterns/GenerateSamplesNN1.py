import random

def generate_samples_labels():
    samples_labels = []
    for _ in range(20000):
        string = ''.join(random.choices(['0', '1'], k=16))
        count_ones = string.count('1')
        label = '1' if count_ones <= 7 else '0'
        samples_labels.append((string, label))

    with open('../testnet1_samples_labels.txt', 'w') as file:
        for sample, label in samples_labels:
            file.write(f"{sample}\t{label}\n")

def split_samples_labels():
    with open('../testnet1_samples_labels.txt', 'r') as file:
        lines = file.readlines()

    samples = [line.split()[0] for line in lines]
    labels = [line.split()[1] for line in lines]

    with open('../testnet1.txt', 'w') as file:
        for sample in samples:
            file.write(f"{sample}\n")

    with open('../testnet1_labels.txt', 'w') as file:
        for label in labels:
            file.write(f"{label}\n")

# Generate the samples and labels file
generate_samples_labels()

# Split the samples and labels into separate files
split_samples_labels()