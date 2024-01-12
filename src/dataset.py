import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizerFast
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 

class BertDataset(Dataset):

    def __init__(self, df, bert_path, num_labels, max_spans):
        self.tok = BertTokenizerFast.from_pretrained(bert_path)
        # labels
        self.labels = list(df['labels'])
        # get encodings and character/token offsets to match start/end characters with token ids
        self.encodings = list(df['sentences'].apply(lambda x: self.tok(x, return_offsets_mapping=True)))
        self.token_ids = [i['input_ids'] for i in self.encodings]
        self.offset_mappings = [i['offset_mapping'] for i in self.encodings]
        # start and end character idxs
        self.all_start_idxs = list(df['start_idxs'])
        self.all_end_idxs = list(df['end_idxs'])
        # number of spans for each sentence
        self.n_spans = list(df['n_spans'])
        # polarity
        self.polarity = list(df['polarity'])
        # total number of labels
        self.num_labels = num_labels
        # maximum number of spans per sentence
        self.max_spans = max_spans

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
      # making labels one-hot tensors
      label = self.labels[idx]
      one_hot_labels = torch.zeros(self.num_labels)
      one_hot_labels[label] = 1
      # polarities 
      pol = self.polarity[idx]
      one_hot_pol = torch.zeros(self.num_labels)
      one_hot_pol[label] = torch.tensor(pol).float() # assigns 1 to negative polarities, 0 otherwise in position of respective labels 
      # making n spans one-hot tensors
      n_spans = self.n_spans[idx]
      one_hot_spans = torch.zeros(self.max_spans+1)
      one_hot_spans[n_spans] = 1
      # ids, offsets, start and end char idxs
      ids = self.token_ids[idx]
      offsets = self.offset_mappings[idx]
      start_char_idxs = self.all_start_idxs[idx]
      end_char_idxs = self.all_end_idxs[idx]

      return one_hot_labels, ids, offsets, start_char_idxs, end_char_idxs, one_hot_spans, one_hot_pol


def get_token_idxs_from_char_idxs(char_idxs, offsets):
  if char_idxs[0]==0 and offsets[1][0]!=0: # when the char range of first word does not start at 0,
    char_idxs[0] = offsets[1][0] # set start_idx on first char of first word in sentence
  mask = [idx for idx, i in enumerate(offsets) if i[1] !=0 and bool(set(char_idxs) & set(range(i[0], i[1]+1)))]
  # if ends==True:
  #    mask = list(np.array(mask)+1)
  return mask


def collate(batch):
  batch_size = len(batch)
  labels = torch.stack([l for l,_,_,_,_,_,_ in batch])
  ids = [i for _,i,_,_,_,_,_ in batch]
  offsets = [o for _,_,o,_,_,_,_ in batch]
  start_char_idxs = [s for _,_,_,s,_,_,_  in batch]
  end_char_idxs = [e for _,_,_,_,e,_,_ in batch]
  n_spans = torch.stack([n for _,_,_,_,_,n,_ in batch])
  pol = torch.stack([p for _,_,_,_,_,_,p in batch])

  max_len = max(len(s) for s in ids) # pad word ids, masks, segs, starts and ends at max len in batch
  sent_pad = torch.zeros((batch_size, max_len), dtype=torch.int64)
  masks_pad = torch.zeros((batch_size, max_len), dtype=torch.int64)
  segs_pad = torch.zeros((batch_size, max_len), dtype=torch.int64)
  starts_pad = torch.zeros((batch_size, max_len), dtype=torch.int64) # full padding is maybe very inefficient...
  ends_pad = torch.zeros((batch_size, max_len), dtype=torch.int64)
  labels_tokens_pad = torch.zeros((batch_size, max_len), dtype=torch.int64)

  for i, s in enumerate(ids):
      sent_pad[i, :len(s)] = torch.tensor(s) # put the embedding in the padded tensor
      masks_pad[i, :len(s)] = 1 # to tell model where the non-padded sentence is
      # leave segs to 0 since not sentence pair classification
      start_token_idxs = get_token_idxs_from_char_idxs(start_char_idxs[i], offsets[i])
      end_token_idxs = get_token_idxs_from_char_idxs(end_char_idxs[i], offsets[i])
      for start, end, label in zip(start_token_idxs, end_token_idxs, torch.nonzero(labels[i])):
        labels_tokens_pad[i, start:end] = label
      starts_pad[i, start_token_idxs] = 1
      ends_pad[i, end_token_idxs] = 1

  return labels, sent_pad, masks_pad, segs_pad, starts_pad, ends_pad, labels_tokens_pad, n_spans, pol 


def calculate_pos_weights(labels, num_labels, one_hot=True):
  if one_hot == False:
    y_train = np.zeros((len(labels), num_labels))
    for idx, i in enumerate(labels):
      y_train[idx, i] = 1 # set indices in relevant row to 1
  else:
    y_train = labels
  num_positives = np.sum(y_train, axis=0)
  num_negatives = len(y_train) - num_positives
  pos_weights  = np.log((num_negatives+1)/(num_positives+1)) # avoid division by zero
  min_weight = min(pos_weights)
  pos_weights = pos_weights/min_weight
  # pos_weights = (num_negatives+1)/(num_positives+1)
  return pos_weights


def create_train_val_test(df_sentences, bert_path, labels_col):
  tok = AutoTokenizer.from_pretrained(bert_path)
  df_sentences['token_ids'] = df_sentences['sentences'].apply(lambda x: tok.encode(x)) # this modified original df but following doesn't
  df_sentences = df_sentences[(df_sentences['token_ids'].str.len() > 3) & (df_sentences['token_ids'].str.len() < 512)] # to remove empty strings and \n\n and above max tokens
  X = df_sentences['sentences']
  y = df_sentences[labels_col]
  columns = ['sentences', 'labels', 'start_idxs', 'end_idxs', 'n_spans', 'polarity']

  X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

  starts_train_val = df_sentences['start_idxs'].loc[X_train_val.index]
  ends_train_val = df_sentences['end_idxs'].loc[X_train_val.index]
  spans_train_val = df_sentences['n_spans'].loc[X_train_val.index]
  polarity_train_val = df_sentences['polarity'].loc[X_train_val.index]
  starts_test = df_sentences['start_idxs'].loc[X_test.index]
  ends_test =  df_sentences['end_idxs'].loc[X_test.index]
  spans_test = df_sentences['n_spans'].loc[X_test.index]
  polarity_test = df_sentences['polarity'].loc[X_test.index]

  df_train_val = pd.DataFrame(list(zip(X_train_val, y_train_val, starts_train_val, ends_train_val, spans_train_val, polarity_train_val)),
                columns = columns)
  X2 = df_train_val['sentences']
  y2 = df_train_val['labels']

  X_train, X_val, y_train, y_val = train_test_split(X2, y2, test_size=0.1, random_state=1)

  starts_train = df_train_val['start_idxs'].loc[X_train.index]
  ends_train = df_train_val['end_idxs'].loc[X_train.index]
  spans_train = df_train_val['n_spans'].loc[X_train.index]
  polarity_train = df_train_val['polarity'].loc[X_train.index]
  starts_val = df_train_val['start_idxs'].loc[X_val.index]
  ends_val = df_train_val['end_idxs'].loc[X_val.index]
  spans_val = df_train_val['n_spans'].loc[X_val.index]
  polarity_val = df_train_val['polarity'].loc[X_val.index]

  df_train = pd.DataFrame(list(zip(X_train, y_train, starts_train, ends_train, spans_train, polarity_train)),
                columns=columns)
  df_val = pd.DataFrame(list(zip(X_val, y_val, starts_val, ends_val, spans_val, polarity_val)),
                columns=columns)
  df_test = pd.DataFrame(list(zip(X_test, y_test, starts_test, ends_test, spans_test, polarity_test)),
                columns=columns)
  counter_val = 0
  counter_test = 0
  for s in df_train['sentences']:
    if s in df_val['sentences']:
        counter==1
    if s in df_test['sentences']:
        counter+=1
  print('n duplicates in train and val:', counter_val)
  print('n duplicates in train and test:', counter_test)

  return df_train, df_val, df_test
