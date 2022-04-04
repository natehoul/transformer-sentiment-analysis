import torch
from transformers import BertTokenizer, BertModel

from util import tokenizer, model

import logging

def get_token_embeddings(sentence):
	marked_text = '[CLS] ' + sentence + ' [SEP]'
	
	tokenized_text = tokenizer.tokenize(marked_text)
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

	segments_ids = [1] * len(tokenized_text)
	
	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensor = torch.tensor([segments_ids])

	with torch.no_grad():
		outputs = model(tokens_tensor, segments_tensor)
		hidden_states = outputs[2]

	token_embeddings = torch.stack(hidden_states, dim=0)
	token_embeddings = torch.squeeze(token_embeddings, dim=1)
	token_embeddings = token_embeddings.permute(1, 0, 2)
	
	token_vecs_sum = []
	for token in token_embeddings:
		sum_vec = torch.sum(token[-4:], dim=0)
		token_vecs_sum.append(sum_vec)
	return token_vecs_sum
