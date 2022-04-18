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
            df['vote'] = df['vote'].fillna(0)
            df = df.dropna()

            if save_pickle:
                pickle_name = path.replace('.json.gz', '.pkl')
                print(f'Saving {pickle_name}')
                df.to_pickle(pickle_name)

        # The top half (or quartile or quintile) highest-rated reviews are considered "helpful" (class 1)
        # The rest are considered "unhelpful" (class 0)

        # Default everything to "unhelpful"
        df['overall'] = 0

        for product in tqdm(df['asin'].unique(), desc="Calculating Helpfulness"):
            product_df = df[df['asin'] == product].sort_values(by=['vote'], ascending=False)
            
            num_reviews_of_this_product = product_df.shape[0]
            midpoint = num_reviews_of_this_product // 5 # 2 for half, 4 for quartile, 5 for quintile (bigger = pickier)
            cutoff_vote_count = product_df.iloc[midpoint]['vote']
            
            # Set all reviews with more helpfullness than the midpoit as "helpful"
            df.loc[(df['asin'] == product) & (df['vote'] > cutoff_vote_count), 'overall'] = 1


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