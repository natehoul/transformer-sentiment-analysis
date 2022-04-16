### BERT Preprocessing ###
# Perform text preprocessing and tokenization
# Mostly copied from the HW4 tutorial

import re
import torch
from util import tokenizer
from tqdm import tqdm




# BERT doesn't need much preprocessing, but we can add more if we need to
# Currently totally unused
def preprocess_text(s):
    # Remove duplicate and trailing whitespace, convert all whitespace to simple space
    s = re.sub(r'\s+', ' ', s).strip()

    return s


# Create a function to tokenize a set of texts
def tokenize(data, token_len):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for review in tqdm(data, desc="Tokenizing"):
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_review = tokenizer.encode_plus(
            text=review,                    # Preprocess sentence (put preprocess_text(review) here if using it)
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=token_len,             # Max length to truncate/pad
            padding='max_length',           # Pad sentence to max length
            truncation=True,                # Truncate to max length
            #return_tensors='pt',           # Return PyTorch tensor (commented out bc of the append below?)
            return_attention_mask=True      # Return attention mask
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_review.get('input_ids'))
        attention_masks.append(encoded_review.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

