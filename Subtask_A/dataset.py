import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import numpy as np
import csv

def prepare_features(seq_1, tokenizer, max_seq_length = 32, 
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


class CVEdatasetA(Dataset):

    def __init__(self, root='../Data/SemEval2020-Task4-Commonsense-Validation-and-Explanation/Training_Data/', maxlen=32):

        #Load data and labels
        print('Getting data from: ', root)
        fa = open(root+'subtaskA_answers.csv')
        fd = open(root+'subtaskA_data.csv')

        self.answers = []
        self.data = []
        for line in fa:
            l = line.split(',')
            self.answers.append((int(l[0]), int(l[1][:-1])))

        datareader = csv.reader(fd)    
        for row in datareader:
            if row[0] != 'id':
                self.data.append((int(row[0]), row[1], row[2]))

        #Initialize the BERT tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)

        self.maxlen = maxlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sent1 = self.data[index][1]
        sent2 = self.data[index][2]
        id_n = self.data[index][0]
        assert id_n == self.answers[index][0]
        answer = self.answers[index][1]
        
        #Construct target labels
        if answer == 1:
            label = torch.tensor([1,0])
        else:
            label = torch.tensor([0,1])

        #Preprocessing the text to be suitable for BERT
        tok_id1_tensor, attn_mask1 = prepare_features(sent1, self.tokenizer) #Tokenize the sentence
        tok_id2_tensor, attn_mask2 = prepare_features(sent2, self.tokenizer) #Tokenize the sentence
        
        # tok1 = ['[CLS]'] + tok1 + ['[SEP]'] #Insering the CLS and SEP token in the beginning and end of the sentence
        # tok2 = ['[CLS]'] + tok2 + ['[SEP]'] #Insering the CLS and SEP token in the beginning and end of the sentence
        
        # if len(tok1) < self.maxlen:
        #     tok1 = tok1 + ['[PAD]' for _ in range(self.maxlen - len(tok1))] #Padding sentences
        # else:
        #     tok1 = tok1[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length
        
        # if len(tok2) < self.maxlen:
        #     tok2 = tok2 + ['[PAD]' for _ in range(self.maxlen - len(tok2))] #Padding sentences
        # else:
        #     tok2 = tok2[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length

        # tok_id1 = self.tokenizer.convert_tokens_to_ids(tok1) #Obtaining the indices of the tokens in the BERT Vocabulary
        # tok_id1_tensor = torch.tensor(tok_id1) #Converting the list to a pytorch tensor
        # tok_id2 = self.tokenizer.convert_tokens_to_ids(tok2) #Obtaining the indices of the tokens in the BERT Vocabulary
        # tok_id2_tensor = torch.tensor(tok_id2) #Converting the list to a pytorch tensor
        
        # #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        # attn_mask1 = (tok_id1_tensor != 0).long()
        # attn_mask2 = (tok_id2_tensor != 0).long()
        
        return tok_id1_tensor, tok_id2_tensor, attn_mask1, attn_mask2, label, id_n

if __name__=='__main__':
    trainset = CVEdatasetA(root="../Data/SemEval2020-Task4-Commonsense-Validation-and-Explanation/Training_Data/")
    valset = CVEdatasetA(root="../Data/SemEval2020-Task4-Commonsense-Validation-and-Explanation/Dev_Data/")
    a,b,c,d,e,f = valset.__getitem__(10)
    print(a.shape,c.shape,e.shape)