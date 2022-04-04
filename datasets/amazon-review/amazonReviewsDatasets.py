import os
import pandas as pd

from torch.utils.data import Dataset

import util.loadData

import util.bertEmbed

class ToolsAndHomeImprovementDataset(Dataset):
	def __init__(self, data_path: str, pkl_path: str=None):
		print('For heaven sake, please print')
		if pkl_path is not None:
			print('Pkl found, loading...')
			df = pd.read_pickle(pkl_path)
		else:
			print('No pkl found, loading dataset')
			df = loadData.getDF(data_path)
		
		print(df.columns)
		self.review_text = df['reviewText']
		self.labels = df['overall']
		print(self.labels.iloc[0])
		self.labels[self.labels < 3] = 0
		self.labels[self.labels == 3] = 1
		self.labels[self.labels > 3] = 2

	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self, idx):
		# Modify this to get the BERT embeddings for the words in the review text and return them as a fixed length token sequence
		return bertEmbed.get_token_embeddings(self.review_text.iloc[idx]), self.labels.iloc[idx]
