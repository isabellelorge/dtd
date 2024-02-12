import argparse
import numpy as np
import pandas as pd
import random
import os
import re
import torch 
from itertools import chain
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from dataset import create_train_val_test, BertDataset, calculate_pos_weights, collate
from models import SpanClassifier, calculate_loss
from utils import create_missing_idxs, create_label_dict
from ast import literal_eval
from datetime import datetime
import os
import json
import logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)

date = datetime.now().strftime('%Y_%m_%d')
time = datetime.now().strftime('%H_%M')

torch.autograd.set_detect_anomaly(True)
d = create_label_dict()

def main():
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_sentence_df', type=str, default = '12_02_2024_final_df.csv', help='Path to df with sentences, start/ends and labels.')
    parser.add_argument('--model_path', type=str, default = f'model_{date}_{time}.pt', help='Path to save model.')
    parser.add_argument('--n_epochs', type=int, default=7, help='Number of epochs.')
    parser.add_argument('--random_seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--lr', type=float, default=3e-05, help='Learning rate.')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--bert_path', type=str, default='bert-base-uncased', help='Bert path to use.')
    parser.add_argument('--test', type=bool, default=True, help='Using test set.')
    parser.add_argument('--level', type=str, default='span', help='Level classification: token, span or sentence.')
    parser.add_argument('--pos_weights', type=bool, default=False, help='Using weighted loss by class.')
    parser.add_argument('--separate_polarity', type=bool, default=False, help='Separate classifier for polarity')
    parser.add_argument('--labels_column', type=str, default='labels', help='which label column to use: labels or merged_labels.')

    args = parser.parse_args()
    df_path = args.path_sentence_df
    sentence_df = pd.read_csv(df_path)
    for col in ['start_idxs', 'end_idxs', 'spans', 'labels' , 'merged_labels', 'polarity']:
        sentence_df[col] = sentence_df[col].apply(lambda x: literal_eval(x))

    sentence_df = create_missing_idxs(sentence_df)
    labels_col = args.labels_column
    
    # to avoid zeros clashing with masked tokens in token-level classification
    if args.level=='token':
        sentence_df[labels_col] = sentence_df[labels_col].apply(lambda x: [i+1 for i in x])

    df_train, df_val, df_test = create_train_val_test(sentence_df, bert_path='bert-base-cased', labels_col = labels_col)

    # eval dataset 
    if args.test == True:
        df_eval = df_test
    else:
        df_eval = df_val
   
    # set random seed and device
    random_seed = args.random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    # datasets
    num_labels = max(sentence_df[labels_col])[0] + 1 
    if args.level == 'token':
        num_labels+=1
    max_spans = sentence_df['n_spans'].max()+1
    train_dataset = BertDataset(df_train, args.bert_path, num_labels=num_labels, max_spans=max_spans)

    eval_dataset = BertDataset(df_eval, args.bert_path, num_labels=num_labels, max_spans=max_spans)

    # positive label weights for imbalanced labels
    label_pos_weights = calculate_pos_weights(df_train['labels'], num_labels=num_labels, one_hot=False)
    n_spans_pos_weights = calculate_pos_weights(df_train['n_spans'], num_labels=max_spans, one_hot=False)
    # if args.level == 'sentence':
    #     pos_weights = args.pos_weights
    # else:
    #     pos_weights = True 
    pos_weights = args.pos_weights

    # print info 
    logging.info(f'seed: {random_seed}')
    logging.info(f'test: {args.test}')
    logging.info(f'most frequent label: {np.argmin(label_pos_weights)}') # this should be 40 (NO ANNOTATION)
    logging.info(f'num labels: {num_labels}')
    logging.info(f'max number of spans: {max_spans}')
    logging.info(f'level: {args.level}')
    logging.info(f'pos_weight: {pos_weights}')
    logging.info(f'separate polarity: {args.separate_polarity}')
    logging.info(f'label pos weights: {label_pos_weights}')
    logging.info(f'labels column: {args.labels_column}')

    label_pos_weights = torch.tensor(label_pos_weights).to(device)
    n_spans_pos_weights = torch.tensor(n_spans_pos_weights).to(device)

    # initialise model and optimizer
    model = SpanClassifier(bert_path=args.bert_path, 
                           batch_size=args.batch_size, 
                           num_labels=num_labels, 
                           dropout_rate=args.dropout_rate, 
                           max_spans=max_spans,
                           level=args.level).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training
    logging.info('Train classifier...')
    for epoch in range(1, args.n_epochs + 1):
        logging.info(f'Training epoch {epoch}')
        logging.info(f'train dataset length: {len(df_train)}')
        logging.info(f'test/validation dataset length: {len(df_eval)}')

        # load and batch train
        train_text_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate)
        eval_text_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate)
        model.train()

        # batch idx 
        batch_idx = 1

        # train on batch
        for text_batch in train_text_dataloader:

            # get batch data
            labels, sents, sents_masks, sents_segs, start_idxs, end_idxs, labels_tokens, n_spans, pol = text_batch
            labels = labels.to(device)
            sents = sents.to(device)
            sents_masks = sents_masks.to(device)
            sents_segs = sents_segs.to(device)
            start_idxs = start_idxs.to(device)
            end_idxs = end_idxs.to(device)
            labels_tokens = labels_tokens.to(device)
            n_spans = n_spans.to(device)
            pol = pol.to(device)

            # zero grad
            optimizer.zero_grad()

            # run model
            start_logits, end_logits, label_logits, n_spans_logits, pol_logits, preds_starts, preds_ends, preds_labels, ordered_preds = model(input_ids=sents, masks=sents_masks, segs=sents_segs,
                                                                                                                               )
            # current metrics
            loss = calculate_loss(start_logits=start_logits, end_logits=end_logits,
                                label_logits=label_logits, n_spans_logits=n_spans_logits, pol_logits=pol_logits,
                                start_idxs=start_idxs, end_idxs=end_idxs, labels=labels, tokens=labels_tokens, n_spans=n_spans, pol=pol, 
                                label_pos_weights=label_pos_weights, n_spans_pos_weights=n_spans_pos_weights, 
                                level=args.level)

            if batch_idx % 100 == 0 and batch_idx != 0:
                logging.info(f'Processed {(batch_idx * args.batch_size)} examples...')
                logging.info(f'BATCH IDX {batch_idx}')
                logging.info(f'LOSS {loss}')

            # backward and step
            loss.backward()
            optimizer.step()
            batch_idx += 1

        # evaluate after each epoch
        logging.info(f'Evaluate classifier epoch: {epoch}..')
        model.eval()

        labels_true = list()
        labels_pred = list()
        starts_true = list()
        starts_pred = list()
        ends_true = list()
        ends_pred = list()
        n_spans_true = list()
        n_spans_pred = list()
        tokens_true = list()
        tokens_pred = list()

        with torch.no_grad():

            # get validation data
            for text_batch in eval_text_dataloader:
                # get batch data
                labels, sents, sents_masks, sents_segs, start_idxs, end_idxs, labels_tokens, n_spans, pol = text_batch
                labels = labels.to(device)
                sents = sents.to(device)
                sents_masks = sents_masks.to(device)
                sents_segs = sents_segs.to(device)
                start_idxs = start_idxs.to(device)
                end_idxs = end_idxs.to(device)
                labels_tokens = labels_tokens.to(device)
                n_spans = n_spans.to(device)
                pol = pol.to(device)

                # run model
                start_logits, end_logits, label_logits, n_spans_logits, pol_logits, preds_starts, preds_ends, preds_labels, ordered_preds = model(input_ids=sents, masks=sents_masks, segs=sents_segs)

                if args.level == 'token':
                    token_preds = torch.argmax(torch.nn.functional.softmax(label_logits, dim=1) , dim=1).cpu().detach().numpy()
                    tokens_cpu = labels_tokens.cpu().detach().numpy()
                    tokens_pred.extend(token_preds)
                    tokens_true.extend(tokens_cpu.flatten())

                elif args.level == 'sentence':
                    # label_preds = torch.round(torch.sigmoid(label_logits).cpu().detach().numpy())
                    sig_preds = torch.sigmoid(label_logits).cpu().detach().numpy()
                    # select max index if threshold not met
                    label_preds = (sig_preds > 0.5).astype(int) 
                    max_indices = np.argmax(sig_preds, axis=1)
                    meets_thresh = label_preds.sum(axis=1) > 0
                    label_preds[~meets_thresh, max_indices[~meets_thresh]] = 1
                    labels_cpu = labels.cpu().detach().numpy()
                    labels_pred.extend(label_preds)
                    labels_true.extend(labels_cpu)
                else:
                   # label preds only for predicted top spans
                    start_preds = preds_starts.cpu().detach().numpy()
                    end_preds = preds_ends.cpu().detach().numpy()
                    label_preds = preds_labels.cpu().detach().numpy()
                    labels_cpu = labels.cpu().detach().numpy()
                    starts_cpu = start_idxs.cpu().detach().numpy()
                    ends_cpu = end_idxs.cpu().detach().numpy()
                    n_span_preds = torch.argmax(n_spans_logits, dim=1).cpu().detach().numpy()
                    n_spans_cpu = torch.argmax(n_spans, dim=1).cpu().detach().numpy()
                    pol_preds = torch.round(torch.sigmoid(pol_logits)).cpu().detach().numpy()
                    pol_cpu = pol.cpu().detach().numpy()

                    if args.separate_polarity == True:
                        # transform back into a 41 labels classification task for evaluation using bitwise masks, this should have shape batch * 41 
                        label_preds = np.concatenate([(label_preds.astype(int)&~pol_preds.astype(int))[:, :-1], (label_preds.astype(int)&pol_preds.astype(int))], axis = 1)
                        labels_cpu = np.concatenate([(labels_cpu.astype(int)&~pol_cpu.astype(int))[:, :-1], (labels_cpu.astype(int)&pol_cpu.astype(int))], axis=1)

                    labels_pred.extend(label_preds)
                    labels_true.extend(labels_cpu)
                    starts_pred.extend(start_preds)
                    starts_true.extend(starts_cpu)
                    ends_pred.extend(end_preds)
                    ends_true.extend(ends_cpu)
                    n_spans_pred.extend(n_span_preds)
                    n_spans_true.extend(n_spans_cpu)
              
            if args.level == 'token':
                print(classification_report(tokens_true, tokens_pred, labels=list(range(1, num_labels)), zero_division=0))
            elif args.level == 'sentence':
                # unique_labels = list(set(np.where(np.array(labels_true)==1)[1]))
                print(classification_report(labels_true, labels_pred, zero_division=0))
            else:
                unique_labels = list(set(np.where(np.array(labels_true)==1)[1]))
                print(classification_report(labels_true, labels_pred, zero_division=0, labels=unique_labels))

            torch.save(model.state_dict(), args.model_path)

if __name__ == "__main__":
    main()