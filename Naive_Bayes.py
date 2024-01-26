import sys
import numpy as np
import pandas as pd

def compute_probability(value, mean, variance):
    exponent = np.exp(-((value - mean) ** 2 / (2 * variance)))
    result = (1 / (np.sqrt(2 * np.pi * variance))) * exponent
    return result

def calculate_variance(data):
    average = calculate_mean(data)
    var = sum([(x - average) ** 2 for x in data]) / float(len(data) - 1)
    return var

def calculate_mean(data):
    mean = np.mean(data)
    return mean

def class_probabilities_computations(summary, data):
    errors = 0
    for i in range(len(data)):
        gauss_col_1_A = compute_probability(data.iloc[i, 1], summary[0][0], summary[0][1])
        gauss_col_2_A = compute_probability(data.iloc[i, 2], summary[0][2], summary[0][3])

        gauss_col_1_B = compute_probability(data.iloc[i, 1], summary[1][0], summary[1][1])
        gaussian_col_2_B = compute_probability(data.iloc[i, 2], summary[1][2], summary[1][3])

        prob_a = summary[0][4] * gauss_col_1_A * gauss_col_2_A
        prob_b = summary[1][4] * gauss_col_1_B * gaussian_col_2_B

        if (prob_a > prob_b) and (data.iloc[i, 0] != 'A'):
            errors += 1
        elif (prob_b > prob_a) and (data.iloc[i, 0] != 'B'):
            errors += 1
        else:
            pass

    for i in range(2):
        for j in range(5):
            if j == 4:
                print(summary[i][j])
            else:
                print(summary[i][j], end=" ")

    print(errors)

if __name__ == "__main__":
    args = ["--data"]
    arg_len = len(sys.argv)
    args_info = []

    for i in range(len(args)):
        for j in range(1, len(sys.argv)):
            if args[i] == sys.argv[j] and sys.argv[j + 1]:
                args_info.append(sys.argv[j + 1])
    
    data = pd.read_csv(args_info[0], header=None)
    filter_values = np.unique(data.iloc[:, 0])
    summarised_datasets = [[], []]
    
    for i in range(len(filter_values)):
        filtered_df = data[data.iloc[:, 0].isin([filter_values[i]])]
        summarised_datasets[i] = (
            calculate_mean(filtered_df[1]),
            calculate_variance(filtered_df[1]),
            calculate_mean(filtered_df[2]),
            calculate_variance(filtered_df[2]),
            len(filtered_df) / len(data)
        )
    
    probabilities = class_probabilities_computations(summarised_datasets, data)
