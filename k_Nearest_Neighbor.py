import numpy as np
import math
import csv
import sys
import getopt


def main():
    data, k = read_command_line_args()
    casebase, ctc = process_csv_data(data)
    errors = classify_and_count_errors(ctc, casebase, k)
    print_results(errors, casebase)
    
def find_k_nearest_neighbors(k, case, casebase):
    nn = np.empty([0, 3])

    for i in range(len(casebase)):
        if len(nn) < k:
            nn = np.vstack((nn, casebase[i]))
            nn = sorted(nn, key=lambda x: euclidean_distance(case, x))
        elif euclidean_distance(case, casebase[i]) < euclidean_distance(case, nn[k - 1]):
            nn = np.vstack((nn, casebase[i]))
            nn = sorted(nn, key=lambda x: euclidean_distance(case, x))
            nn = nn[:k]

    return nn if len(nn) == k else []

def calculate_weight_i(k, i, case, nn):
    d_k = euclidean_distance(case, nn[k - 1])
    d_1 = euclidean_distance(case, nn[0])

    if d_k == d_1:
        return 1
    else:
        d_i = euclidean_distance(case, nn[i])
        return (d_k - d_i) / (d_k - d_1)
    
def calculate_weights(k, case, nn):
    weights = np.array([calculate_weight_i(k, i, case, nn) for i in range(k)])
    return weights

def euclidean_distance(a, b):
    return math.sqrt((a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

def read_command_line_args():
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "", ["data=", "k="])
    except getopt.GetoptError:
        print("Usage: student.py --data <file> --k <value>")
        sys.exit(2)

    data, k = None, None

    for opt, arg in opts:
        if opt == "--data":
            data = arg
        elif opt == "--k":
            k = int(arg)

    if data is None or k is None:
        print("Usage: student.py --data <file> --k <value>")
        sys.exit(2)

    return data, k

def process_csv_data(data):
    casebase, ctc = np.empty([0, 3], dtype=float), np.empty([0, 3], dtype=float)

    with open(data) as f:
        reader = csv.reader(f)
        for row in reader:
            row[0] = 1 if row[0] == "A" else 2
            row[1], row[2] = float(row[1]), float(row[2])

            if len(casebase) == 0 or find_k_nearest_neighbors(1, row, casebase)[0][0] != row[0]:
                casebase = np.vstack([casebase, row])
            else:
                ctc = np.vstack([ctc, row])

    return casebase, ctc

def classify_and_count_errors(ctc, casebase, k):
    errors = 0

    for c in ctc:
        c_class, cweights = 0, np.zeros(2)
        nn = find_k_nearest_neighbors(k, c, casebase)
        w = calculate_weights(k, c, nn)

        for i in range(k):
            if nn[i][0] == 1:
                cweights[0] += w[i]
            else:
                cweights[1] += w[i]

        c_class = 1 if cweights[0] > cweights[1] else 2

        if c_class != c[0]:
            errors += 1

    return errors

def print_results(errors, casebase):
    print(errors)

    for case in casebase:
        res = "A" if case[0] == 1 else "B"
        res += f"\t{case[1]}\t{case[2]}"
        print(res)

def main():
    data, k = read_command_line_args()
    casebase, ctc = process_csv_data(data)
    errors = classify_and_count_errors(ctc, casebase, k)
    print_results(errors, casebase)

if __name__ == "__main__":
    main()
