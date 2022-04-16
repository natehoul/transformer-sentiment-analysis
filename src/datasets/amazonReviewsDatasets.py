import os
import pandas as pd

from torch.utils.data import Dataset

from util import loadData, bertEmbed
from raw_data import get_raw_data
from datasets import selected_columns


# TODO (?): Make Dataset for helpfulness ratings (this is a wholly separate project task, and can wait / may not be needed)
	# This maybe shouldn't be in the datasets/amazon-review directory, but instead teh datasets/review_helpfulness dir (?)
# TODO: Use logging instead of printing to stdout; ideally global logging for the whole program (init in some __init__.py file somewhere)
class AmazonReviewsDataset(Dataset):
	def __init__(self, dataset_name: str, save_pickle: bool=True, binarize=True):
		is_pickle, path, size = get_raw_data(dataset_name)

		if is_pickle:
			print(f'Loading {os.path.basename(path)}')
			df = pd.read_pickle(path)
		else:
			df = loadData.getDF(path, size)
			df = df[selected_columns].dropna()
			
			if save_pickle:
				pickle_name = path.replace('.json.gz', '.pkl')
				print(f'Saving {pickle_name}')
				df.to_pickle(pickle_name)

		self.review_text = df['reviewText'].copy()
		self.labels = df['overall'].copy()
		self.labels[self.labels < 3] = 0
		self.labels[self.labels == 3] = 0 if binarize else 1
		self.labels[self.labels > 3] = 1 if binarize else 2

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
	dataset = AmazonReviewsDataset('tools')