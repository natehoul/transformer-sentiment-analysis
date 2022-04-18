import os
import pandas as pd

from tqdm import tqdm

from torch.utils.data import Dataset

from util import loadData, bertEmbed
from raw_data import get_raw_data


# If you ever change this (such as to add 'vote'), delete all .pkl files from raw_data
# Also PLEASE keep this consistet with the version in helpfulnessDatasess.py
selected_columns = ['reviewText', 'overall', 'asin', 'vote']


class HelpfulnessDataset(Dataset):
    def __init__(self, dataset_name: str, save_pickle: bool=True, binarize=True):
        assert binarize, "Multiclass Helpfull Not Currently Supported for Helpfullness"
        is_pickle, path, size = get_raw_data(dataset_name)

        if is_pickle:
            print(f'Loading {os.path.basename(path)}')
            df = pd.read_pickle(path)
        else:
            df = loadData.getDF(path, size)
            df = df[selected_columns]
            df['vote'] = df['vote'].str.replace(',', '').fillna(0).astype(int)
            df = df.dropna()

            if save_pickle:
                pickle_name = path.replace('.json.gz', '.pkl')
                print(f'Saving {pickle_name}')
                df.to_pickle(pickle_name)

        # The overwhelming majority of reviews have 0 helpfulness votes (~85% of the music instruments dataset)
        # Just classify reviews with ANY helpfulness votes as begin helpful and the rest unhelpful
        df['overall'] = 0
        df.loc[df['vote'] > 0, 'overall'] = 1

        self.review_text = df['reviewText'].copy()
        self.labels = df['overall'].copy()
            
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Modify this to get the BERT embeddings for the words in the review text and return them as a fixed length token sequence
        # NOTE / CONCERN: If we perform the embedding here (instead of in __init__), this may result in a significant slow-down during training
        # There's also a LOT of repeated work in each epoch
        # Also if there's a problem with the embeddings, we won't know until we start training
        # Correction: It just occurred to me that the embeddings change slightly as the embedding model trains, so redoing the work is necessary
        return bertEmbed.get_token_embeddings(self.review_text.iloc[idx]), self.labels.iloc[idx]


    def get_all_data(self):
        return self.review_text.to_numpy(), self.labels.to_numpy()



# For testing stuff
if __name__ == "__main__":
    dataset = HelpfulnessDataset('music')