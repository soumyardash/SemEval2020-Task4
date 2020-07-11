import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import numpy as np
import csv

def prepare_features(seq_1, tokenizer, max_seq_length = 64, 
             zero_pad = True, include_CLS_token = True, include_SEP_token = True):
    ## Tokenzine Input
    tokens_a = tokenizer.tokenize(seq_1)

    ## Truncate
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    ## Initialize Tokens
    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)
    ## Add Tokens and separators
    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    ## Input Mask 
    input_mask = [1] * len(input_ids)
    ## Zero-pad sequence lenght
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    return torch.tensor(input_ids), torch.tensor(input_mask)

class CVEdatasetB(Dataset):

    def __init__(self, root='../Data/SemEval2020-Task4-Commonsense-Validation-and-Explanation/Training_Data/', maxlen=64):

        #Load data and labels
        print('Getting data from: ', root)
        fa = open(root+'subtaskB_answers.csv')
        fd = open(root+'subtaskB_data.csv')
        self.answers = []
        self.data = []
        c2l = {'A':0, 'B':1, 'C':2}
        
        ra = csv.reader(fa)
        for row in ra:
            if row[0] == 'id':
              continue
            id_n = int(row[0])
            label = int(c2l[row[1]])
            self.answers.append((id_n, label))

        rd = csv.reader(fd)    
        for row in rd:
            if row[0] == 'id':
              continue
            id_n = int(row[0])
            sen = str(row[1])
            exp1 = str(row[2])
            exp2 = str(row[3])
            exp3 = str(row[4])
            
            if sen[-1] == '.':
              sen = sen[:-1]
            sen = sen + ' is against commonsense because '
            if exp1 != '' and exp1[-1] == '.':
              exp1 = exp1[:-1]
            if exp2 != '' and exp2[-1] == '.':
              exp2 = exp2[:-1]
            if exp3 != '' and exp3[-1] == '.':
              exp3 = exp3[:-1]
          
            self.data.append((id_n, sen, exp1, exp2, exp3))

        #Initialize the BERT tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

        self.maxlen = maxlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sent = self.data[index][1]
        exp1 = self.data[index][2]
        exp2 = self.data[index][3]
        exp3 = self.data[index][4]
        id_n = self.data[index][0]
        
        assert id_n == self.answers[index][0]
        answer = self.answers[index][1]
        
        #Construct target labels
        label = torch.eye(3)[answer]

        #Preprocessing the text to be suitable for BERT
        tok_id1_tensor, attn_mask1 = prepare_features(sent+exp1, self.tokenizer) #Tokenize the sentence
        tok_id2_tensor, attn_mask2 = prepare_features(sent+exp2, self.tokenizer) #Tokenize the sentence
        tok_id3_tensor, attn_mask3 = prepare_features(sent+exp3, self.tokenizer) #Tokenize the sentence
        
  
        # tok1 = ['[CLS]'] + tok1 + ['[SEP]']
        # tok2 = ['[CLS]'] + tok2 + ['[SEP]']
        # tok3 = ['[CLS]'] + tok3 + ['[SEP]']
        
        # if len(tok1) < self.maxlen:
        #     tok1 = tok1 + ['[PAD]' for _ in range(self.maxlen - len(tok1))] #Padding sentences
        # else:
        #     tok1 = tok1[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length
        
        # if len(tok2) < self.maxlen:
        #     tok2 = tok2 + ['[PAD]' for _ in range(self.maxlen - len(tok2))] #Padding sentences
        # else:
        #     tok2 = tok2[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length

        # if len(tok3) < self.maxlen:
        #     tok3 = tok3 + ['[PAD]' for _ in range(self.maxlen - len(tok3))] #Padding sentences
        # else:
        #     tok3 = tok3[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length

        # tok_id1 = self.tokenizer.convert_tokens_to_ids(tok1) #Obtaining the indices of the tokens in the BERT Vocabulary
        # tok_id1_tensor = torch.tensor(tok_id1) #Converting the list to a pytorch tensor
        # tok_id2 = self.tokenizer.convert_tokens_to_ids(tok2) #Obtaining the indices of the tokens in the BERT Vocabulary
        # tok_id2_tensor = torch.tensor(tok_id2) #Converting the list to a pytorch tensor
        # tok_id3 = self.tokenizer.convert_tokens_to_ids(tok3) #Obtaining the indices of the tokens in the BERT Vocabulary
        # tok_id3_tensor = torch.tensor(tok_id3) #Converting the list to a pytorch tensor
        
        # #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        # attn_mask1 = (tok_id1_tensor != 0).long()
        # attn_mask2 = (tok_id2_tensor != 0).long()
        # attn_mask3 = (tok_id3_tensor != 0).long()
        
        return tok_id1_tensor, tok_id2_tensor, tok_id3_tensor, attn_mask1, attn_mask2, attn_mask3, label, id_n
    
if __name__=='__main__':
    trainset = CVEdatasetB(root="../Data/SemEval2020-Task4-Commonsense-Validation-and-Explanation/Training_Data/")
    valset = CVEdatasetB(root="../Data/SemEval2020-Task4-Commonsense-Validation-and-Explanation/Dev_Data/")
    a,b,c,d,e,f,g,h = trainset.__getitem__(1000)
    print(a.shape,b.shape,c.shape,d.shape,e.shape,f.shape,g.shape)