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


class SpanClassifier(nn.Module):
    def __init__(self, batch_size, num_labels, bert_path, dropout_rate, max_spans):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
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

    def forward(self, input_ids, masks, segs):
        # batch * tokens * dim => batch * tokens
        all_span_logits = []

        outputs = self.bert(input_ids, attention_mask=masks, token_type_ids=segs)
        tokens_rep = self.dropout(outputs.last_hidden_state) # should also try mean pooling this to see if performs better?
        seq_rep = self.dropout(outputs.pooler_output)

        start_logits = self.start_classifier(tokens_rep).squeeze(-1)
        end_logits = self.end_classifier(tokens_rep).squeeze(-1)
        spans_logits = self.span_classifier(seq_rep)
        n_spans_logits = self.span_number_classifier(seq_rep)

        all_n_spans = torch.argmax(n_spans_logits, dim=1) # predicted number of spans FOR EACH EXAMPLE IN BATCH (batch_size*max_n_spans+1)

        preds_labels = []
        preds_starts = []
        preds_ends = []
        for idx, n_spans in enumerate(all_n_spans):
          spans = spans_logits[idx] # if unique label, classify full sentence
          starts = start_logits[idx]
          ends = end_logits[idx]
          labels_onehot = torch.zeros_like(spans)
          starts_onehot = torch.zeros_like(starts)
          ends_onehot = torch.zeros_like(ends) # should classify only span, ie use start/end?

          if n_spans <=1: # also extract span when unique?
            # Create one-hot encoding
            labels_onehot[torch.argmax(spans)] = 1
            starts_onehot[torch.argmax(starts)] = 1
            ends_onehot[torch.argmax(ends)] = 1
            preds_labels.append(labels_onehot)
            preds_starts.append(starts_onehot)
            preds_ends.append(ends_onehot)

          else:
            top_n_starts, top_n_ends = get_top_spans(starts, ends, n_spans)
            if len(top_n_starts) < n_spans:
              print('FEWER VALID SPANS THAN PREDICTED NUMBER')
            span_vectors = [] # get span vectors 
            for idx, start in enumerate(top_n_starts):
              end = top_n_ends[idx] + 1 # Add 1 since end index is exclusive
              starts_onehot[start] = 1
              ends_onehot[start] = 1
              span_vector = tokens_rep[idx, start:end, :].mean(dim=0) # get relevant tokens
              span_vectors.append(span_vector)
            preds_starts.append(starts_onehot) # need to addd one list per sentence/data point 
            preds_ends.append(ends_onehot)

            # label span vectors
            if span_vectors:
              span_vecs = torch.stack(span_vectors) # create tensor 
              span_logits = self.span_classifier(span_vecs) # to classify all vecs simultaneously
              span_labels = torch.argmax(span_logits, dim =1)  # get predictions of each span for inference 
              labels_onehot.scatter_(0, span_labels, 1)
              preds_labels.append(labels_onehot)
              spans,_ = torch.max(span_logits, dim=0)  # TAKING THE MAX IS BEST OPTION (it gets the logit from where the span was correctly predicted if it was anywhere)


            else:
              print('NO VALID SPANS')
              span_logits = torch.zeros(self.num_labels) # dummy probabilities in case there are no valid spans

          all_span_logits.append(spans)  # this should be batch * labels
          span_logits_tensor = torch.stack(all_span_logits) # create tensor 
          preds_starts_tensor = torch.stack(preds_starts) 
          preds_ends_tensor = torch.stack(preds_ends) 
          preds_labels_tensor = torch.stack(preds_labels) 
        return start_logits, end_logits, span_logits_tensor, n_spans_logits, preds_starts_tensor, preds_ends_tensor, preds_labels_tensor


def calculate_loss(start_logits, end_logits, span_logits, n_spans_logits,
                   start_idxs, end_idxs, labels, n_spans,
                   label_pos_weights, n_spans_pos_weights):
  
  crit_starts = nn.BCEWithLogitsLoss()
  crit_ends = nn.BCEWithLogitsLoss()
  crit_labels = nn.BCEWithLogitsLoss(pos_weight=label_pos_weights)
  # crit_spans = nn.CrossEntropyLoss(weight=n_spans_pos_weights)
  crit_spans = nn.CrossEntropyLoss()

  starts_loss = crit_starts(start_logits, start_idxs.float()) # start_probs and end_probs have size batch * n_tokens
  ends_loss = crit_ends(end_logits, end_idxs.float())
  labels_loss = crit_labels(span_logits, labels.float()) # labels has size batch * labels
  n_spans_loss = crit_spans(n_spans_logits, n_spans.float()) # n_spans has size batch * max_n_spans + 1

  loss = starts_loss + ends_loss + labels_loss + n_spans_loss
  return loss