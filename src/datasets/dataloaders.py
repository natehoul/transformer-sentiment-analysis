# Here's the deal
# All we *really* care about is the dataloaders
# The data-*sets* are only EVER used in service of the creation of the data loaders
# Why bother dealing with the middle man?
# Much of this is inspired by the tutorial from HW4

from torch import tensor
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import processed_data
from raw_data import resolve_alias
from amazonReviewsDatasets import AmazonReviewsDataset
from util.data_manipulation import create_validation_split, balance_data
from util.bert_preprocessing import tokenize


# Given a dataset name and some specification about how you like your data, do the following:
#   Load the raw data (using AmazonReviewsDataset)
#       (This will need to change somewhat when we do helpfulness)
#   Split the data into training and validation
#   Balance the data so there's an equal number of each class
def load_split_balance_save(dataset_name: str, binarize=True, val_split=0.2, save_data=True):
    dataset_name = resolve_alias(dataset_name)
    dataset = AmazonReviewsDataset(dataset_name=dataset_name, binarize=binarize)
    data, labels = dataset.get_all_data()
    
    tx, ty, vx, vy = create_validation_split(data, labels, val_split)
    tx, ty = balance_data(tx, ty)
    vx, vy = balance_data(vx, vy)

    if save_data:
        processed_data.save(dataset_name, tx, ty, binarize=binarize, is_val=False)
        processed_data.save(dataset_name, vx, vy, binarize=binarize, is_val=True, val_split=val_split)

    return tx, ty, vx, vy


# Assembles a dataloader from the inputs (the data), masks (used to train BERT), and the labels (for the classifier)
def generate_dataloader(inputs, masks, labels, shuffle, batch_size):
    dataset = TensorDataset(inputs, masks, labels)
    
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader


# This is the one to use!
# Input: The name of a dataset (any valid alias from raw_data)
# Output: balanced, processed, tokenized dataloaders for training and validation
def get_dataloaders_end_to_end(dataset_name, binarize=True, val_split=0.2, save_data=True, batch_size=32, token_len=64):
    dataset_name = resolve_alias(dataset_name)

    # Tokenized data is available, so just load it directly
    if processed_data.exists(dataset_name, binarize=binarize, is_val=True, val_split=val_split, tokenized=True):
        t_input, t_mask, t_label = processed_data.load_tokens(dataset_name, binarize=binarize, is_val=False, val_split=val_split, token_len=token_len)
        v_input, v_mask, v_label = processed_data.load_tokens(dataset_name, binarize=binarize, is_val=True, val_split=val_split, token_len=token_len)


    # Tokenized data not available, must tokenize
    else:
        # Attempt to load data that has already been split and balanced
        if processed_data.exists(dataset_name, binarize=binarize, is_val=True, val_split=val_split, tokenized=False):
            tx, ty = processed_data.load(dataset_name, binarize=binarize, is_val=False, val_split=val_split)
            vx, vy = processed_data.load(dataset_name, binarize=binarize, is_val=True, val_split=val_split)
        
        # Data doesn't exist in processed_data at all (no balancing, no val split); load from scratch
        else:
            tx, ty, vx, vy = load_split_balance_save(dataset_name, binarize=binarize, val_split=val_split, save_data=save_data)
    
        # Perform tokenization
        t_label = tensor(ty)
        t_input, t_mask = tokenize(tx, token_len)

        v_label = tensor(vy)
        v_input, v_mask = tokenize(vx, token_len)

        # Save tokens for later
        if save_data:
            processed_data.save_tokens(dataset_name, t_input, t_mask, t_label, binarize=binarize, is_val=False, val_split=val_split, token_len=token_len)
            processed_data.save_tokens(dataset_name, v_input, v_mask, v_label, binarize=binarize, is_val=True, val_split=val_split, token_len=token_len)


    t_loader = generate_dataloader(t_input, t_mask, t_label, shuffle=True, batch_size=batch_size)
    v_loader = generate_dataloader(v_input, v_mask, v_label, shuffle=False, batch_size=batch_size)

    return t_loader, v_loader
    


# For testing the code
if __name__ == "__main__":
    tl, vl = get_dataloaders_end_to_end('music')


    for i, (id, mask, label) in enumerate(vl):
        print(id, mask, label, sep='\n\n', end='\n\n\n\n')

        if i > 10:
            break