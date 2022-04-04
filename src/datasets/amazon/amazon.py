
import pandas as pd
import gzip
import json

import os

AMAZON_REVIEW_DF_PATH = '../../../datasets/amazon-review/amazon-review-df.pkl'
AMAZON_REVIEW_DATA_PATH = '../../../datasets/amazon-review/Books_5.json.gz'

def parse(path):
	g = gzip.open(path, 'rb')
	for l in g:
		yield json.loads(l)

def getDF(path):
	if os.path.exists(os.path.join(os.path.realpath(__file__), AMAZON_REVIEW_DF_PATH)):
		return pd.read_pickle(os.path.join(os.path.realpath(__file__), AMAZON_REVIEW_DF_PATH))
	
	i = 0
	df = {}
	for d in parse(path):
		df[i] = d
		i += 1
	return pd.DataFrame.from_dict(df, orient='index')

def read_data():
	df = getDF(AMAZON_REVIEW_DATA_PATH)
	return df
