
import pandas as pd
import numpy as np
import argparse

def root_entropy(ds):
    row_count = len(ds)
    label = len(ds.columns) - 1
    unique_target = ds[label].unique()
    tot_entropy = 0

    for class_value in unique_target:
        class_count= ds[ds[label] == class_value].shape[0]
        entropy = - (class_count / row_count) * (np.log(class_count / row_count) / np.log(c))
        tot_entropy+= entropy

    print(f"0,root,{tot_entropy},no_leaf")

def calculate_info_gain(ds):
    attribute_count = len(ds.columns) - 1
    label = attribute_count
    unique_target = ds[label].unique()
    entropy_list = [0] * attribute_count
    total_count = ds.shape[0]

    for i in range(attribute_count):
        attribute_values = ds[i].unique()
        gain_entropy = 0

        for value in attribute_values:
            value_ds = ds[ds[i] == value]
            value_count = value_ds.shape[0]
            tot_entropy= 0

            for class_value in unique_target:
                value_label_count = value_ds[value_ds[label] == class_value].shape[0]
                if value_label_count != 0:
                    entropy = - (value_label_count / value_count) * (np.log(value_label_count / value_count) / np.log(c))
                    tot_entropy+= entropy

            gain_entropy += (tot_entropy* (value_count / total_count))

        entropy_list[i] = gain_entropy

    high_info_index = entropy_list.index(min(entropy_list))
    return high_info_index

def non_root_entropies(ds, attr, depth):
    attribute_values = ds[attr].unique()
    label = len(ds.columns) - 1

    pure_class_values = []

    for value in attribute_values:
        value_data = ds[ds[attr] == value]
        value_total = value_data.shape[0]
        unique_target = value_data[label].unique()
        tot_entropy= 0

        for class_value in unique_target:
            value_count = value_data[value_data[label] == class_value].shape[0]
            if value_count == len(value_data):
                entropy = 0
                pure_class_values.append(value)
            else:
                entropy = - (value_count / value_total) * (np.log(value_count / value_total) / np.log(c))
            tot_entropy+= entropy

            if value in pure_class_values:
                class_assigned = class_value
            else:
                class_assigned = "no_leaf"

        print(f"{depth},att{attr}={value},{tot_entropy},{class_assigned}")

    return pure_class_values

def id3_decision_tree(ds, attribute_value_pairs, depth):
    if attribute_value_pairs is None:
        attribute_value_pairs = []

        for col in ds.columns[:-1]:
            for val in ds[col].unique():
                attribute_value_pairs.append([col, val])

    if len(attribute_value_pairs) == 0:
        return

    attr_index = calculate_info_gain(ds)
    pure_values = non_root_entropies(ds, attr_index, depth)
    filtered_attribute_value_pairs = [[i, j] for i, j in attribute_value_pairs if not (i == attr_index and j in pure_values)]
    attribute_value_pairs = filtered_attribute_value_pairs
    split_values = ds[attr_index].unique()

    if len(pure_values) != 0:
        for value in pure_values:
            ds_filtered = ds[ds[attr_index] != value]
        ds = ds_filtered

    for value in split_values:
        if value not in pure_values:
            sub_ds = ds[ds[attr_index] == value]
            id3_decision_tree(sub_ds, attribute_value_pairs, depth=depth + 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    args = parser.parse_args()
    data_file = args.data

    ds = pd.read_csv(data_file, header=None)
    c = ds[ds.columns[-1]].nunique()

    root_entropy(ds)
    attribute_value_pairs = None
    id3_decision_tree(ds, attribute_value_pairs,depth=1)