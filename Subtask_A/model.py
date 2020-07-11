import torch
import torch.nn as nn
from transformers import RobertaModel

class CVEclassifier(nn.Module):

    def __init__(self, freeze_roberta = False, hidden_dropout_prob=0.1, num_labels=1):
        super(CVEclassifier, self).__init__()
        #Instantiating BERT model object 
        self.roberta_layer = RobertaModel.from_pretrained('roberta-large')
        
        #Freeze bert layers
        if freeze_roberta:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        #Dropout
        self.dropout = nn.Dropout(hidden_dropout_prob)
        #Classifier layer
        #We are predicting scores for a sentence
        # self.fc = nn.Linear(768, 256)
        self.classifier = nn.Linear(1024, num_labels)
        
    def forward(self, tok_id1_tensor, tok_id2_tensor, attn_mask1, attn_mask2):
        #Feeding the input to BERT model to obtain contextualized representations
        bert_hidden_states1, _ = self.roberta_layer(tok_id1_tensor, attention_mask = attn_mask1)
        bert_hidden_states2, _ = self.roberta_layer(tok_id2_tensor, attention_mask = attn_mask2)
        

        #Extract [CLS] embeddings
        sent_emb1 = bert_hidden_states1[:,0]
        sent_emb2 = bert_hidden_states2[:,0]

        # sent_1 = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)
        # sent_emb1 = bert_hidden_states1.mean(1)
        # sent_emb2 = bert_hidden_states2.mean(1)
        
        #Calculate sentence scores/logit
        logit1 = self.classifier(self.dropout(sent_emb1))
        logit2 = self.classifier(self.dropout(sent_emb2))

        #Concatenate to get full logits
        logits = torch.cat((logit1, logit2), 1)

        return logits
