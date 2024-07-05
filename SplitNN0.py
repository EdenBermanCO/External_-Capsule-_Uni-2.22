import numpy as np

default_input_txt = 'nn0.txt'


def split_file(file, train_file, test_file, test_ratio=0.2):
    with open(file, "r") as f:
        data = f.readlines()

    # shuffle the data
    np.random.shuffle(data)

    # calculate the test set size
    test_set_size = int(len(data) * test_ratio)
    train_data = data[test_set_size:]
    test_data = data[:test_set_size]

    # write train data to file
    with open(train_file, "w") as train_f:
        train_f.writelines(train_data)

    # write test data to file
    with open(test_file, "w") as test_f:
        test_f.writelines(test_data)

    print("Data split and saved successfully!")

split_file(default_input_txt, 'train_nn0.txt', 'test_nn0.txt')