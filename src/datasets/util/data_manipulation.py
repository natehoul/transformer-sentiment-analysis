# A small suite of functions for managing some data manipulation operations

from random import sample
import numpy as np


# Input: 2 numpy arrays, one representing a class label
# Output: The same arrays, with some values dropped such that the labels are balanced
def balance_data(data, labels):
    # Count how many instances there are of each class
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.setdefault(label, 0) + 1

    classes = class_counts.keys()
    min_count = min(class_counts.values())

    # Identify the locations (indices) of the data for each class
    class_indices = {}
    for c in classes:
        class_indices[c] = np.argwhere(labels == c).flatten()

    # Randomly select a fixed number of data (identified via their index) to use
    # Since the same number is used for each class, this balances the data
    balanced_indices_by_class = {}
    for c in classes:
        balanced_indices_by_class[c] = np.random.choice(class_indices[c], min_count)

    # Combine all the selected indices from the various classes into one big set of 
    # all the indices we want from the whole dataset
    balanced_indices = np.concatenate(list(balanced_indices_by_class.values()))
    balanced_indices.sort()

    # Return only the selected, balanced data and corresponding labels
    return data[balanced_indices], labels[balanced_indices]



# Input: 2 numpy arrays of equal length
# Output  4 numpy arrays representing a training/validation split
def create_validation_split(data, labels, val_split=0.20):
    assert val_split <= 0.8, "Validation split too large"
    assert val_split >= 0.05, "Validation split too small"

    # Determine how much data there is, and how much of it is training data
    num_entries = len(data)
    num_training = int(num_entries * (1 - val_split))
    
    # Determine which data samples (identified by their indices) will be used for 
    # training, and which will be used for testing
    # Use sets to leverage set difference
    all_indicies = set(range(num_entries))
    training_indices = set(sample(all_indicies, num_training))
    validation_indices = all_indicies - training_indices

    assert len(training_indices & validation_indices) == 0, "Training and validation sets not disjoint"
    assert len(training_indices | validation_indices) == num_entries, "Training and validation sets do not encompass all data"

    # Convert to lists to work with numpy indexing
    training_indices = sorted(training_indices)
    validation_indices = sorted(validation_indices)

    # Select out the data via index to be used for training and testing data and labels
    tx, ty = data[training_indices], labels[training_indices]
    vx, vy = data[validation_indices], labels[validation_indices]

    return tx, ty, vx, vy



# Converts a number representing a class to an array
# i.e. 0 --> [1, 0, 0 ,0]; 1 --> [0, 1, 0, 0]
def create_class_vector(c, num_classes):
    v = [0] * num_classes
    v[int(c)] = 1
    return v


# Apply create_class_vector to an array of class labels
# input is a numpy ndarray
def convert_labels_to_vectors(labels):
    num_classes = len(np.unique(labels))
    new_labels = [create_class_vector(l, num_classes) for l in labels]
    return np.array(new_labels)
