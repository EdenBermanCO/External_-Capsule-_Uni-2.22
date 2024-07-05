def check_pattern(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    count_satisfied = 0
    count_not_satisfied = 0

    for line in lines:
        string, label = line.split()
        if label == '1':
            count_ones = string.count('1')
            if count_ones >= 8 and count_ones <=12:
                count_satisfied += 1
            else:
                count_not_satisfied += 1

    print(f"Pattern satisfied: {count_satisfied} strings")
    print(f"Pattern not satisfied: {count_not_satisfied} strings")

# Usage example
check_pattern('../nn0.txt')
