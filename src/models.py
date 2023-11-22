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
  # Create joint probabilities matrix 
  joint_probs = start_probs[:, None] * end_probs[None, :]

  # Mask out invalid locations (where end before start, ie lower triangle)
  mask = torch.tril(torch.ones_like(joint_probs)) == 0
  joint_probs = joint_probs * mask

  # Greedily select top non-overlapping spans
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
          start_logits = end_logits = n_spans_logits = preds_starts_tensor = preds_ends_tensor = preds_labels_tensor = torch.tensor(0) # placeholders
        # sentence-level classifier 
        elif self.level == 'sentence':
          label_logits = full_seq_logits
          # n_spans_logits = self.span_number_classifier(seq_rep)
          n_spans_logits = start_logits = end_logits = preds_starts_tensor = preds_ends_tensor = preds_labels_tensor = torch.tensor(0) # placeholders
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

          for idx, n_spans in enumerate(all_n_spans): # better way than looping n of spans..? extracting indices of 0/1s..?
            spans = full_seq_logits[idx]
            starts = start_logits[idx]
            ends = end_logits[idx]
            labels_onehot = torch.zeros_like(spans)
            starts_onehot = torch.zeros_like(starts)
            ends_onehot = torch.zeros_like(ends) 

            if n_spans <=1: # if unique label, classify full sentence
              # Create one-hot encoding to get discrete labels for inference
              labels_onehot[torch.argmax(spans)] = 1
              starts_onehot[torch.argmax(starts)] = 1
              ends_onehot[torch.argmax(ends)] = 1
              preds_labels.append(labels_onehot)
              preds_starts.append(starts_onehot)
              preds_ends.append(ends_onehot)

            else:
              # we transform the variable 'spans' to replace full seq
              top_n_starts, top_n_ends = get_top_spans_nongreedy(starts, ends, n_spans)
              # if n_spans < 4:
              #   print('TRUE', torch.nonzero(start_idxs[idx]), torch.nonzero(end_idxs[idx]), torch.argmax(true_n_spans[idx]))
              #   print('PAIRS', list(zip(top_n_starts, top_n_ends)))
              #   if len(top_n_starts) < n_spans:
              #     print('NOT ENOUGH VALID SPANS')

              span_vectors = [] # get span vectors 
              for i, start in enumerate(top_n_starts):
                # discrete predictions for inference
                end = top_n_ends[i]
                starts_onehot[start] = 1
                ends_onehot[end] = 1
                # get relevant tokens for span classification
                span_vector = tokens_rep[idx, start:end, :].mean(dim=0) 
                span_vectors.append(span_vector)
              # inference
              preds_starts.append(starts_onehot)
              preds_ends.append(ends_onehot)

              # label span vectors
              if span_vectors:
                span_vecs = torch.stack(span_vectors) # create tensor 
                multiple_span_logits = self.span_classifier(span_vecs) # to classify all vecs simultaneously
                # discrete predictions for inference 
                span_labels = torch.argmax(multiple_span_logits, dim =1)  
                labels_onehot.scatter_(0, span_labels, 1)
                preds_labels.append(labels_onehot)
                # TAKING THE MAX IS BEST OPTION (it gets the logit from where the span was correctly predicted if it was anywhere)
                spans,_ = torch.max(multiple_span_logits, dim=0)  
              else:
                print('NO VALID SPANS')
                spans = torch.zeros(self.num_labels) # dummy probabilities for labels in case there are no valid spans

            all_span_logits.append(spans)  # this should be batch * labels

          label_logits = torch.stack(all_span_logits) # create tensor
          # discrete predictions for inference 
          preds_starts_tensor = torch.stack(preds_starts) 
          preds_ends_tensor = torch.stack(preds_ends) 
          preds_labels_tensor = torch.stack(preds_labels)
        
        return start_logits, end_logits, label_logits, n_spans_logits, preds_starts_tensor, preds_ends_tensor, preds_labels_tensor


def calculate_loss(start_logits, end_logits, label_logits, n_spans_logits,
                   start_idxs, end_idxs, labels, tokens, n_spans,
                   label_pos_weights, n_spans_pos_weights, level, use_pos_weight):
  tokens = tokens.view(-1)
  crit_starts = nn.BCEWithLogitsLoss()
  crit_ends = nn.BCEWithLogitsLoss()
  crit_tokens = nn.CrossEntropyLoss()
  crit_spans = nn.CrossEntropyLoss()
  # crit_spans = nn.CrossEntropyLoss(weight=n_spans_pos_weights)
  if use_pos_weight == True:
    crit_labels = nn.BCEWithLogitsLoss(pos_weight=label_pos_weights)
  else:
     crit_labels = nn.BCEWithLogitsLoss()
  if level == 'token':
    labels_loss = crit_tokens(label_logits, tokens)
    starts_loss = ends_loss = n_spans_loss = 0 # placeholders 
  elif level == 'sentence':
    labels_loss = crit_labels(label_logits, labels.float())
    # n_spans_loss = crit_spans(n_spans_logits, n_spans.float())
    starts_loss = ends_loss = n_spans_loss = 0 # placeholders 
  else:
    labels_loss = crit_labels(label_logits, labels.float()) # labels has size batch * labels
    starts_loss = crit_starts(start_logits, start_idxs.float()) # start_probs and end_probs have size batch * n_tokens
    ends_loss = crit_ends(end_logits, end_idxs.float())
    n_spans_loss = crit_spans(n_spans_logits, n_spans.float()) # n_spans has size batch * max_n_spans + 1

  loss = starts_loss + ends_loss + labels_loss + n_spans_loss

  return loss