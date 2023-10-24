import argparse
import numpy as np
import pandas as pd
import random
import os
import torch 
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from dataset import create_train_val_test, BertDataset, calculate_pos_weights, collate
from models import SpanClassifier, calculate_loss
from utils import create_missing_idxs
from ast import literal_eval

torch.autograd.set_detect_anomaly(True)

def main():

    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_sentence_df', type=str, default='/mnt/sdg/isabelle/dtd_project/data/annotations/span_level_annot/processed_sentences_labels/simplified.csv', help='Path to df with sentences, start/ends and labels.')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--random_seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--lr', type=float, default=3e-05, help='Learning rate.')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--bert_path', type=str, default='bert-base-cased', help='Bert path to use.')
    parser.add_argument('--num_labels', type=int, default=41, help='Number of labels.')
    parser.add_argument('--max_spans', type=int, default=10, help='Maximum possible number of spans per sentence.')
    parser.add_argument('--test', type=bool, default=False, help='Using test set.')

    args = parser.parse_args()
    sentence_df = pd.read_csv(args.path_sentence_df)
    sentence_df['start_idxs'] = sentence_df['start_idxs'].apply(lambda x: literal_eval(x))
    sentence_df['end_idxs'] = sentence_df['end_idxs'].apply(lambda x: literal_eval(x))
    sentence_df['spans'] = sentence_df['spans'].apply(lambda x: literal_eval(x))
    sentence_df['labels'] = sentence_df['labels'].apply(lambda x: literal_eval(x))
    sentence_df = create_missing_idxs(sentence_df)
    sentence_df['labels_unique'] = sentence_df['labels'].apply(lambda x: list(set(x)))
 
    # print(sentence_df['labels'])
  
    df_train, df_val, df_test = create_train_val_test(sentence_df, bert_path='bert-base-cased')
    random_seed = args.random_seed
    print('SEED:', random_seed, 'TEST:', args.test)
   
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # datasets
    train_dataset = BertDataset(df_train, args.bert_path, num_labels=args.num_labels)
    eval_dataset = BertDataset(df_val, args.bert_path, num_labels=args.num_labels)

    # positive label weights for imbalanced labels
    label_pos_weights = calculate_pos_weights(df_train['labels'], num_labels=args.num_labels, one_hot=False)
    print('most frequent label', np.argmin(label_pos_weights)) # this should be 40 (no label)
    label_pos_weights = torch.tensor(label_pos_weights).to(device)
    n_spans_pos_weights = calculate_pos_weights(df_train['n_spans'], num_labels=args.max_spans+1, one_hot=False)
    n_spans_pos_weights = torch.tensor(n_spans_pos_weights).to(device)

    # initialise model and optimizer
    model = SpanClassifier(bert_path=args.bert_path, batch_size=args.batch_size, num_labels=args.num_labels, dropout_rate=args.dropout_rate).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training
    print('Train classifier...')
    for epoch in range(1, args.n_epochs + 1):
        print(f'Training epoch {epoch}')
        print(df_train['sentences'].head())
        print(f'train dataset length: {len(df_train)}')
        print(df_val['sentences'].head())
        print(f'test/validation dataset length: {len(df_val)}')

        # load and batch train (! maybe balance batch instead of weighting loss?)
        train_text_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate)
        eval_text_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate)

        model.train()
        # metrics 
        batch_idx = 1
        train_loss = 0
        train_labels_f1 = 0
        train_starts_f1 = 0
        train_ends_f1 = 0
        train_spans_f1 = 0


        # train on batch
        for text_batch in train_text_dataloader:

            # get batch data
            labels, sents, sents_masks, sents_segs, start_idxs, end_idxs, n_spans = text_batch
            labels = labels.to(device)
            sents = sents.to(device)
            sents_masks = sents_masks.to(device)
            sents_segs = sents_segs.to(device)
            start_idxs = start_idxs.to(device)
            end_idxs = end_idxs.to(device)
            n_spans = n_spans.to(device)

            # zero grad
            optimizer.zero_grad()

            # run model
            start_logits, end_logits, span_logits, n_spans_logits = model(input_ids=sents, masks=sents_masks, segs=sents_segs)

            # current metrics
            loss = calculate_loss(start_logits=start_logits, end_logits=end_logits,
                                span_logits=span_logits, n_spans_logits=n_spans_logits,
                                start_idxs=start_idxs, end_idxs=end_idxs, labels=labels, n_spans=n_spans,
                                label_pos_weights=label_pos_weights, n_spans_pos_weights=n_spans_pos_weights)

            # predictions
            label_preds = torch.round(torch.sigmoid(span_logits)).cpu().detach().numpy()
            start_preds = torch.round(torch.sigmoid(start_logits)).cpu().detach().numpy()
            end_preds = torch.round(torch.sigmoid(end_logits)).cpu().detach().numpy()
            span_preds = torch.round(torch.sigmoid(n_spans_logits)).cpu().detach().numpy()
            span_preds_arg = torch.argmax(n_spans_logits, dim=1).cpu().detach().numpy()

            # cpu ground truths
            labels_cpu = labels.cpu().detach().numpy()
            starts_cpu = start_idxs.cpu().detach().numpy()
            ends_cpu = end_idxs.cpu().detach().numpy()
            spans_cpu = n_spans.cpu().detach().numpy()

            # F1 scores (get precision and recall?)
            labels_f1m = f1_score(labels_cpu, label_preds, average = 'macro', zero_division=0)
            labels_precision_recall = precision_recall_fscore_support(labels_cpu, label_preds, average = 'macro', zero_division=0)
            starts_f1m = f1_score(starts_cpu, start_preds, average = 'macro', zero_division=0)
            ends_f1m = f1_score(ends_cpu, end_preds, average = 'macro', zero_division=0)
            spans_f1m =  f1_score(spans_cpu, span_preds, average = 'macro', zero_division=0)

            train_loss+=loss.item()

            train_labels_f1+=labels_f1m.item()
            train_starts_f1+=starts_f1m.item()
            train_ends_f1+=ends_f1m.item()
            train_spans_f1+=spans_f1m.item()

            if batch_idx % 10 == 0 and batch_idx != 0:
                print(f'Processed {(batch_idx * args.batch_size)} examples...')
                print('BATCH IDX', batch_idx)
                print('LOSS', loss)
                print('AVG_LOSS:', train_loss/batch_idx)
                print('prec recal labels', labels_precision_recall)
                print('AVG_labels_F1:', train_labels_f1/batch_idx)
                print('AVG_starts_F1:', train_starts_f1/batch_idx)
                print('AVG_ends_F1:', train_ends_f1/batch_idx)
                print('AVG_spans_F1:', train_spans_f1/batch_idx)

                print('LABEL PREDS', np.nonzero(label_preds[0]), np.nonzero(labels[0]))
                print('START PREDS', np.nonzero(start_preds[0]), np.nonzero(start_idxs[0]))
                print('END PREDS', np.nonzero(end_preds[0]), np.nonzero(end_idxs[0]))
                print('N SPANS PREDS', span_preds_arg, torch.argmax(n_spans, dim=1))

            # backward and step
            loss.backward()
            optimizer.step()
            batch_idx += 1

        # evaluate after each epoch
        print(f'Evaluate classifier epoch: {epoch}..')
        model.eval()

        labels_true = list()
        labels_pred = list()
        starts_true = list()
        starts_pred = list()
        ends_true = list()
        ends_pred = list()
        spans_true = list()
        spans_pred = list()

        with torch.no_grad():

            # get validation data
            for text_batch in eval_text_dataloader:
                # get batch data
                labels, sents, sents_masks, sents_segs, start_idxs, end_idxs, n_spans = text_batch
                labels = labels.to(device)
                sents = sents.to(device)
                sents_masks = sents_masks.to(device)
                sents_segs = sents_segs.to(device)
                start_idxs = start_idxs.to(device)
                end_idxs = end_idxs.to(device)
                n_spans = n_spans.to(device)

                # run model
                start_logits, end_logits, span_logits, n_spans_logits = model(input_ids=sents, masks=sents_masks, segs=sents_segs)

                # current metrics
                loss = calculate_loss(start_logits=start_logits, end_logits=end_logits,
                                        span_logits=span_logits, n_spans_logits=n_spans_logits,
                                        start_idxs=start_idxs, end_idxs=end_idxs, labels=labels, n_spans=n_spans,
                                        label_pos_weights=label_pos_weights, n_spans_pos_weights=n_spans_pos_weights)

                label_preds = torch.round(torch.sigmoid(span_logits)).cpu().detach().numpy()
                start_preds = torch.round(torch.sigmoid(start_logits)).cpu().detach().numpy()
                end_preds = torch.round(torch.sigmoid(end_logits)).cpu().detach().numpy()
                span_preds = torch.argmax(n_spans_logits, dim=1)

                labels_cpu = labels.cpu().detach().numpy()
                starts_cpu = start_idxs.cpu().detach().numpy()
                ends_cpu = end_idxs.cpu().detach().numpy()
                spans_cpu = n_spans.cpu().detach().numpy()

                labels_pred.extend(label_preds)
                labels_true.extend(labels_cpu)
                starts_pred.extend(start_preds)
                starts_true.extend(starts_cpu)
                ends_pred.extend(end_preds)
                ends_true.extend(ends_cpu)
                spans_pred.extend(span_preds)
                spans_true.extend(spans_cpu)

            unique_labels = list(set(np.where(np.array(labels_true)==1)[1]))
            print(classification_report(labels_true, labels_pred, zero_division=0, labels=unique_labels))


if __name__ == "__main__":
    main()