# A suit of functions for managing the files in processed_data
# tbh probably shouldn't be in __init__.py but rather some other file but whatever

import os
import numpy as np
import torch


# Procedurally create a filename based on the details of the data
def generate_filename(name, binarize, is_val, val_split, tokenized, token_len=0, is_data=True) -> str:
    b = '_binary' if binarize else ''
    v = f'_validation_{int(val_split*100)}' if is_val else f'_training_{int((1-val_split)*100)}'
    d = '_data' if is_data else '_labels'
    t = f'_tokens{d}_{token_len}' if tokenized else ''
    ext = '.pt' if tokenized else '.npy'
    basename = f'{name}{b}{v}{t}{ext}'
    filename = f'{os.path.dirname(__file__)}/{basename}'
    return filename



# Check if a file already exists for some data, based on the details of that data
def exists(name, binarize=True, is_val=False, val_split=0.2, tokenized=False, token_len=64) -> bool:
    filename = generate_filename(name, binarize, is_val, val_split, tokenized, token_len)
    return os.path.isfile(f'{filename}')



# Save data and labels for later
# Data and labels are np arrays for data that has not yet been tokenized
def save(name, data, labels, binarize=True, is_val=False, val_split=0.2):
    filename = generate_filename(name, binarize, is_val, val_split, tokenized=False)
    combined = np.stack((data, labels))
    np.save(filename, combined, allow_pickle=True)



# Load data and labels as np arrays
# Data and labels are np arrays for data that has not yet been tokenized
def load(name, binarize=True, is_val=False, val_split=0.2):
    filename = generate_filename(name, binarize, is_val, val_split, tokenized=False)
    combined = np.load(filename, allow_pickle=True)
    data, labels = combined[0], combined[1].astype(np.float32)
    return data, labels



# Save the tokenized data used by BERT
# Inputs, masks, and labels are pytorch tensors, not np arrays
def save_tokens(name, inputs, masks, labels, binarize=True, is_val=False, val_split=0.2, token_len=64):
    data_filename = generate_filename(name, binarize, is_val, val_split, tokenized=True, token_len=token_len, is_data=True)
    labels_filename = generate_filename(name, binarize, is_val, val_split, tokenized=True, token_len=token_len, is_data=False)
    data = torch.stack((inputs, masks))
    torch.save(data, data_filename)
    torch.save(labels, labels_filename)


# Load the tokenized data for use by BERT
# This way, we only have to perform the (costly) tokenization process once for a given data spec
# Inputs, masks, and labels are pytorch tensors, not np arrays
def load_tokens(name, binarize=True, is_val=False, val_split=0.2, token_len=64):
    data_filename = generate_filename(name, binarize, is_val, val_split, tokenized=True, token_len=token_len, is_data=True)
    labels_filename = generate_filename(name, binarize, is_val, val_split, tokenized=True, token_len=token_len, is_data=False)
    data = torch.load(data_filename)
    labels = torch.load(labels_filename)
    inputs, masks = data[0], data[1]
    return inputs, masks, labels

