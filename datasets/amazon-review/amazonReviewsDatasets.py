import os
import pandas as pd

from torch.utils.data import Dataset

from util import loadData, bertEmbed
from raw_data import get_raw_data


selected_columns = ['reviewText', 'overall']

# TODO (?): Make Dataset for helpfulness ratings (this is a wholly separate project task, and can wait / may not be needed)
	# This maybe shouldn't be in the datasets/amazon-review directory, but instead teh datasets/review_helpfulness dir (?)
# TODO: Use logging instead of printing to stdout; ideally global logging for the whole program (init in some __init__.py file somewhere)
class AmazonReviewsDataset(Dataset):
	def __init__(self, dataset_name: str, save_pickle: bool=True):
		is_pickle, path, size = get_raw_data(dataset_name)

		if is_pickle:
			print(f'Loading {os.path.basename(path)}')
			df = pd.read_pickle(path)
		else:
			df = loadData.getDF(path, size)
			df = df[selected_columns]
			
			if save_pickle:
				pickle_name = path.replace('.json.gz', '.pkl')
				print(f'Saving {pickle_name}')
				df.to_pickle(pickle_name)

		self.review_text = df['reviewText'].copy()
		self.labels = df['overall'].copy()
		self.labels[self.labels < 3] = 0
		self.labels[self.labels == 3] = 1
		self.labels[self.labels > 3] = 2

	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self, idx):
		# Modify this to get the BERT embeddings for the words in the review text and return them as a fixed length token sequence
		# NOTE / CONCERN: If we perform the embedding here (instead of in __init__), this may result in a significant slow-down during training
		# There's also a LOT of repeated work in each epoch
		return bertEmbed.get_token_embeddings(self.review_text.iloc[idx]), self.labels.iloc[idx]
