# Code for saving/loading pytorch models
# The only reason why I put this here is so that I can use __file__ to make sure
# the models get saved into the correct directory
# It's also consistent with raw_data and processed_data

import os
import torch
from training.results import TIMESTAMP

# All models saved here have a filename in this format
filename_auto_stamp = f'{os.path.dirname(__file__)}/{TIMESTAMP}_{"{}"}.pt'
filename_complete = f'{os.path.dirname(__file__)}/{"{}"}.pt'

# Check if a file exists
def exists(name):
    return os.path.isfile(filename_complete.format(name))


# Save a model with a specified name
# The name *should* be descriptive of the model's hyperparameters or 
# the data it was trained on, but it doesn't strictly have to be
def save(model, name):
    torch.save(model, filename_auto_stamp.format(name))


# Load the model with the name specified in save()
def load(name):
    return torch.load(filename_complete.format(name))