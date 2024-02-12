import torch 
from torch import nn
from transformers import BertModel

def overlaps(start, selected_starts, selected_ends):
  # Return True if (start, end) overlaps with any previous spans
  for s, e in zip(selected_starts, selected_ends):
    if start <= e: # if new start is before any previously selected end
      return True
  return False


def get_top_spans(start_probs, end_probs, n_spans):
  # create joint probabilities matrix 
  joint_probs = start_probs[:, None] * end_probs[None, :]

  # mask out invalid locations (where end before start, ie lower triangle)
  mask = torch.tril(torch.ones_like(joint_probs)) == 0
  joint_probs = joint_probs * mask

  # greedily select top non-overlapping spans
  selected_starts, selected_ends = [], []

  while len(selected_starts) < n_spans:
    # check if any valid spans remaining
    if joint_probs.max() == 0:
      break
    # Get max joint index
    max_idx = torch.argmax(joint_probs)
    
    start = max_idx // end_probs.shape[0] # this gives row (ie if matrix M*N, new row every N indices)
    end = max_idx % end_probs.shape[0] # remainder from modulo gives column (how far down N)
    # Check for no overlap
    if not overlaps(start, selected_starts, selected_ends):
      # Add to selected
      selected_starts.append(start)
      selected_ends.append(end)

    # Mask out updated span
    joint_probs[start,:] = 0 # ie, all combinations with this start
    joint_probs[:,end] = 0 # all combinations with this end 

  return selected_starts, selected_ends


def get_top_spans_nongreedy(starts, ends, n_spans):
    # non-greedy version to avoid always selecting start-end of sentence if combined confidence is highest
    n_spans_copy = n_spans.clone()
    v_s, i_s = torch.topk(torch.sigmoid(starts), k=n_spans_copy)
    v_e, i_e = torch.topk(torch.sigmoid(ends), k=n_spans_copy)
    top_starts_sorted, top_starts_idxs = torch.sort(i_s)
    top_ends_sorted, top_ends_idxs = torch.sort(i_e)
    top_starts = []
    top_ends = []
    s_idx = 0
    e_idx = 0
    last_end = 0
    while n_spans_copy > 0 and s_idx < n_spans and e_idx < n_spans:
        s = top_starts_sorted [s_idx]
        e = top_ends_sorted [e_idx]
        if e > s and s >= last_end:
            top_starts.append(s)
            top_ends.append(e)
            last_end = e
            n_spans_copy-=1
            s_idx+=1
            e_idx+=1
        elif e <=s :
            e_idx+=1
        else:
            s_idx+=1
    return top_starts, top_ends 


class SpanClassifier(nn.Module):
    def __init__(self, batch_size, num_labels, bert_path, dropout_rate, max_spans, level):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        # token classifier
        self.token_classifier = nn.Linear(768, num_labels)
        # start and end classifiers
        self.start_classifier = nn.Linear(768, 1)
        self.end_classifier = nn.Linear(768, 1)
        # classifier to predict number of spans
        self.span_number_classifier = nn.Linear(768, max_spans+1)
        # classifier to predict span label
        self.span_classifier = nn.Linear(768, num_labels)
        # classifier to predict negative polarity (concatenating predicted labels to BERT sequence output)
        self.neg_pol_classifier = nn.Linear(768 + num_labels, num_labels)
        # dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        # num labels
        self.num_labels = num_labels
        # batch size
        self.batch_size = batch_size
        # level for classification (token, sentence or span)
        self.level = level

    def forward(self, input_ids, masks, segs):
        # batch * tokens * dim => batch * tokens
        outputs = self.bert(input_ids, attention_mask=masks, token_type_ids=segs)
        tokens_rep = self.dropout(outputs.last_hidden_state) # each token
        seq_rep = self.dropout(outputs.pooler_output) # full sentence/sequence
        full_seq_logits = self.span_classifier(seq_rep)
       
        # token-level classifier
        if self.level == 'token':
          label_logits = self.token_classifier(tokens_rep).view(-1, self.num_labels)
          start_logits = end_logits = n_spans_logits = pol_logits = preds_starts_tensor = preds_ends_tensor = preds_labels_tensor = torch.tensor(0) # placeholders
        # sentence-level classifier 
        elif self.level == 'sentence':
          label_logits = full_seq_logits
          # n_spans_logits = self.span_number_classifier(seq_rep)
          start_logits = end_logits =  n_spans_logits = pol_logits = preds_starts_tensor = preds_ends_tensor = preds_labels_tensor = torch.tensor(0) # placeholders
        # span-level classifier 
        else:
          all_span_logits = [] 
          start_logits = self.start_classifier(tokens_rep).squeeze(-1)
          end_logits = self.end_classifier(tokens_rep).squeeze(-1)
          n_spans_logits = self.span_number_classifier(seq_rep)
          all_n_spans = torch.argmax(n_spans_logits, dim=1) # predicted number of spans FOR EACH EXAMPLE IN BATCH (batch_size*max_n_spans+1)
          # all_n_spans = torch.count_nonzero(torch.round(torch.sigmoid(full_seq_logits)), dim = 1)
          preds_labels = []
          preds_starts = []
          preds_ends = []
          all_ordered_preds = []

          for idx, n_spans in enumerate(all_n_spans): # looping all predicted span numbers
            spans = nn.functional.softmax(full_seq_logits[idx], dim=-1)
            starts = start_logits[idx]
            ends = end_logits[idx]
            labels_onehot = torch.zeros_like(spans)
            starts_onehot = torch.zeros_like(starts)
            ends_onehot = torch.zeros_like(ends)
            ordered_preds = []

            if n_spans <=1: # if unique label, classify full sentence
              # Create one-hot encoding to get discrete labels for inference
              label = torch.argmax(spans)
              confidence = torch.max(spans)
              start = torch.argmax(starts)
              end = torch.argmax(ends)
              labels_onehot[label] = 1
              starts_onehot[start] = 1
              ends_onehot[end] = 1
              preds_labels.append(labels_onehot)
              preds_starts.append(starts_onehot)
              preds_ends.append(ends_onehot)
              ordered_preds.append((start, end, label, confidence))

            else: # if more than one span predicted in sentence
              top_n_starts, top_n_ends = get_top_spans_nongreedy(starts, ends, n_spans) # get top start/end pairs in order of confidence
              span_vectors = [] # get span vectors 
              for i, start in enumerate(top_n_starts):
                # discrete start/end predictions for inference
                end = top_n_ends[i]
                starts_onehot[start] = 1
                ends_onehot[end] = 1
                # get relevant tokens for span classification
                span_vector = tokens_rep[idx, start:end+1, :].mean(dim=0)
                soft = nn.functional.softmax(self.span_classifier(span_vector.unsqueeze(0)), dim=1)
                confidence = torch.max(soft)
                label = torch.argmax(soft)
                ordered_preds.append((start, end, label, confidence))
                span_vectors.append(span_vector)

              # label span vectors
              if span_vectors:
                span_vecs = torch.stack(span_vectors) # create tensor 
                # this is just n_spans*labels because within each sample
                multiple_span_logits = nn.functional.softmax(self.span_classifier(span_vecs), dim=1) # to classify all vecs simultaneously
                # discrete label predictions for inference 
                span_labels = torch.argmax(multiple_span_logits, dim =1)  
                labels_onehot.scatter_(0, span_labels, 1)
                # max pooling 
                spans,_ = torch.max(multiple_span_logits, dim=0)  
              else:
                print('NO VALID SPANS', n_spans, top_n_starts, top_n_ends)
                labels_onehot[torch.argmax(spans)] = 1
                spans = torch.zeros(self.num_labels).to(input_ids.device) # dummy probabilities for labels in case there are no valid spans
              
              # inference
              preds_starts.append(starts_onehot)
              preds_ends.append(ends_onehot)
              preds_labels.append(labels_onehot)

            all_span_logits.append(spans)  # this should be batch * labels
            all_ordered_preds.append(ordered_preds)

          label_logits = torch.stack(all_span_logits) # create tensor
          # polarity
          pol_logits = self.neg_pol_classifier(torch.concat([seq_rep, label_logits], dim = 1)) # should be batch size * seq dim + num labels
          # discrete predictions for inference 
          preds_starts_tensor = torch.stack(preds_starts) 
          preds_ends_tensor = torch.stack(preds_ends) 
          preds_labels_tensor = torch.stack(preds_labels)
        
        return start_logits, end_logits, label_logits, n_spans_logits, pol_logits, preds_starts_tensor, preds_ends_tensor, preds_labels_tensor, all_ordered_preds


def calculate_loss(start_logits, end_logits, label_logits, n_spans_logits,
                   pol_logits,
                   start_idxs, end_idxs, labels, tokens, n_spans, pol,
                   label_pos_weights, n_spans_pos_weights, level):
  tokens = tokens.view(-1)
  crit_starts = nn.BCEWithLogitsLoss()
  crit_ends = nn.BCEWithLogitsLoss()
  crit_tokens = nn.CrossEntropyLoss()
  crit_spans = nn.CrossEntropyLoss()
  crit_pol = nn.BCEWithLogitsLoss()
  crit_labels = nn.BCELoss(reduction='none')
     
  if level == 'token':
    labels_loss = crit_tokens(label_logits, tokens)
    starts_loss = ends_loss = n_spans_loss = pol_loss = 0 # placeholders 
  elif level == 'sentence':
    crit_labels = nn.BCEWithLogitsLoss()
    labels_loss = crit_labels(label_logits, labels.float())
    starts_loss = ends_loss = n_spans_loss = pol_loss = 0 # placeholders 
  else:
    # implementing the weighted loss myself since cannot use logits loss
    labels_loss = crit_labels(label_logits, labels.float())
    labels_loss*=label_pos_weights # labels has size batch * labels
    starts_loss = crit_starts(start_logits, start_idxs.float()) # start_probs and end_probs have size batch * n_tokens
    ends_loss = crit_ends(end_logits, end_idxs.float())
    n_spans_loss = crit_spans(n_spans_logits, n_spans.float()) # n_spans has size batch * max_n_spans + 1
    pol_loss = crit_pol(pol_logits, pol.float())

  loss = starts_loss + ends_loss + labels_loss.mean() + n_spans_loss + pol_loss 
    
  return loss