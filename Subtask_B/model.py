import torch
import torch.nn as nn
from transformers import RobertaModel

class CVEclassifier(nn.Module):

    def __init__(self, freeze_bert = False, hidden_dropout_prob=0.15, num_labels=1):
        super(CVEclassifier, self).__init__()
        #Instantiating BERT model object 
        self.bert_layer = RobertaModel.from_pretrained('roberta-base')
        
        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        #Dropout
        self.dropout = nn.Dropout(hidden_dropout_prob)
        #Classifier layer
        #We are predicting scores for a sentence
        self.classifier = nn.Linear(768, num_labels)
        
    def forward(self, tok_id1_tensor, tok_id2_tensor, tok_id3_tensor, attn_mask1, attn_mask2, attn_mask3):
        #Feeding the input to BERT model to obtain contextualized representations
        bert_hidden_states1, _ = self.bert_layer(tok_id1_tensor, attention_mask = attn_mask1)
        bert_hidden_states2, _ = self.bert_layer(tok_id2_tensor, attention_mask = attn_mask2)
        bert_hidden_states3, _ = self.bert_layer(tok_id3_tensor, attention_mask = attn_mask3)
        
        #Extract [CLS] embeddings
        sent_emb1 = bert_hidden_states1[:,0]
        sent_emb2 = bert_hidden_states2[:,0]
        sent_emb3 = bert_hidden_states3[:,0]

        #Calculate sentence scores/logit
        logit1 = self.classifier(self.dropout(sent_emb1))
        logit2 = self.classifier(self.dropout(sent_emb2))
        logit3 = self.classifier(self.dropout(sent_emb3))
        
        #Concatenate to get full logits
        logits = torch.cat((logit1, logit2, logit3), 1)
        
        return logits
