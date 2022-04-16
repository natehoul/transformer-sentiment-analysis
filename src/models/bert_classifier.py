# This is very similar to BertClassifier from HW4
# Basically, user BertModel.from_pretrained to embed the input tokens,
# Then use a classifer on the embedded tokens
# Main difference is that the classifer is much more sophisticated

from pprint import pprint
import sys

import torch.nn as nn
from transformers import BertModel
from transformer import TransformerEncoder



# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, num_classes: int = 2, num_layers: int = 6, dim_model: int = 512, num_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(BertClassifier, self).__init__()
        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate a transformer-based classifer
        self.classifier = TransformerEncoder(output_size=num_classes, 
                                            num_layers=num_layers,
                                            dim_model=dim_model,
                                            num_heads=num_heads,
                                            dim_feedforward=dim_feedforward,
                                            dropout=dropout)

        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        # TBH I'm not entirely sure what this does or why it works
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


if __name__ == "__main__":
    print("hello")