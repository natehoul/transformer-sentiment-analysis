# This is very similar to BertClassifier from HW4
# Basically, user BertModel.from_pretrained to embed the input tokens,
# Then use a classifer on the embedded tokens
# Main difference is that the classifer is much more sophisticated

import torch
import torch.nn as nn
from transformers import BertModel
from transformer import TransformerEncoder



# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, num_classes: int = 2, num_layers: int = 6, dim_model: int = 768, num_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1, use_advanced_model=False):
        super(BertClassifier, self).__init__()
        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased') #, output_hidden_states=True)

        # Instantiate a transformer-based classifer
        if use_advanced_model:
            self.classifier = TransformerEncoder(output_size=num_classes, 
                num_layers=num_layers,
                dim_model=dim_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
                )

        else:
            self.classifier = nn.Sequential(
            nn.Linear(dim_model, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, num_classes)
            )

        self.use_advanced_model = use_advanced_model

        
    def forward(self, tokens_tensor, attention_mask):
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

        if self.use_advanced_model:
            segments_ids = [1] * tokens_tensor.size(1)
    
            segments_tensor = torch.tensor([segments_ids]).to(tokens_tensor.get_device())


            with torch.no_grad():
                outputs = self.bert(tokens_tensor, segments_tensor) # NOTE: Error here. See belohidden_states = outputs[2]
                hidden_states = outputs[2]

            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1, 0, 2, 3)
            token_vecs_sum = []
            for token in token_embeddings:
                sum_vec = torch.sum(token[-4:], dim=0)        
                token_vecs_sum.append(sum_vec)
            #print(token_vecs_sum.shape)
            embeddings = torch.stack(token_vecs_sum).to(tokens_tensor.get_device())
            print('EMBEDDING SIZE')
            print(embeddings.size())
            #embedding = torch.tensor([token_vecs_sum]).to(tokens_tensor.get_device())
            #print(embedding.size())

        else:
            outputs = self.bert(input_ids=tokens_tensor,
                                attention_mask=attention_mask)
  
            # Extract the last hidden state of the token `[CLS]` for classification task
            last_hidden_state_cls = outputs[0][:, 0, :]
            embeddings = last_hidden_state_cls
                


        # Feed input to classifier to compute logits
        logits = self.classifier(embeddings)

        return logits


if __name__ == "__main__":
    print("hello")
