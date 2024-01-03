import pandas as pd
import spacy
import re
import random
from ast import literal_eval

nlp = spacy.load("en_core_web_sm")

targets = ['older_age', 'family_member_mental_disorder', ':abuse', 'childhood_abuse', 'long_illness_duration', 'severe_illness',
           'suicidality', 'multiple_hospitalizations', 'recurrent_episodes', 'improvement',
           'physical_comorbidity', 'mental_comorbidity', 'substance_abuse', 'anhedonia', 'illness_early_onset',
           'multiple_antidepressants', 'antidepressant_dosage_increase', 'multiple_psychotherapies', 'side_effects', 'non_adherence']

def create_label_dict(level='span'):
    # create integers/labels dictionary
    targets_polar =([f'{t}_POSITIVE' for t in targets]+[f'{t}_NEGATIVE' for t in targets])
    targets_polar.append('NO_ANNOTATION')
    if level=='token':
       d = {k:i for k,i in list(zip(targets_polar, list(range(1, len(targets_polar)+1))))}
    else:
       d = {k:i for k,i in list(zip(targets_polar, list(range(0, len(targets_polar)))))}
    return d


def put_label_inside(note):
  # put labels back inside sentence when outside (period immediately followed by label)
  labels_outside = re.finditer('\. *\\n* *\[.*?\(.*?TIVE\).*?\]', note)
  for i, l in enumerate(labels_outside):
    clean_label = re.sub('\.', '', l.group()) + '.'
    note = re.sub(re.escape(l.group()), clean_label, note)
  return note


def process_annot_df(annot_df):
  # process raw notes df: those that contain correct labels, put back labels inside sentences, add ID for note
  annot_df = annot_df[annot_df['notes'].str.contains('\[.*?\(.*?TIVE\).*?\]')].reset_index(drop=True) # remove notes without labels
  annot_df['notes']= annot_df['notes'].apply(lambda x: put_label_inside(x)) # put the labels back into sentence rather than after
  annot_df['id'] = [f'NOTE_{i}' for i in range(len(annot_df))]
  return annot_df


def clean_df(sentence_df):
  # remove rows where labels parsed ended with non-coherent start/ends
  bad_rows = []
  for idx, starts in sentence_df['start_idxs'].items():
    ends = sentence_df['end_idxs'].iloc[idx]
    for idx2, start in enumerate(starts):
      end = ends[idx2]
      if end <= start:
        bad_rows.append(idx)
        print(idx, 'end before start', starts, ends)
      if len(starts) > idx2 + 1 and starts[idx2+1] < end:
        bad_rows.append(idx)
        print(idx, 'next start before end', starts, ends)
  return sentence_df.drop(bad_rows)


def get_label(label_text, regex, d):
    # get label for annotation (needs labels dictionary). Choose randomly if multiple labels for a span. 
    l = []
    label_text = re.sub('hospitalisations', 'hospitalizations', label_text)
    label_text = re.findall(regex, label_text)[0]
    for t in targets:
      if t in label_text or re.sub('_', ' ', t) in label_text or t.upper() in label_text:
        if 'POSITIVE' in label_text:
          # positive annot
          l.append( d[f'{t}_POSITIVE'])
        else :
           # negative annot
          l.append(d[f'{t}_NEGATIVE'])
    if l != []:
      l = random.choice(l)
    return l


def get_spans(sentence, d):
  # divide sentences into spans, get start/ends and labels 
  start_idxs = []
  end_idxs = []
  spans_text = []
  labels = []
  start = 0
  regex = re.compile('\[.*?\(.*?TIVE\).*?\].?')
  # clean sentence
  clean_sentence = re.sub(regex, '', sentence).strip()
  clean_sentence = re.sub(r'\s{2,}', ' ', clean_sentence)
  # find all labels in sentence
  spans_idxs = [i.span() for i in re.finditer(regex, sentence)]
  for span in spans_idxs:
    clean_span = re.sub(regex, '', sentence[start:span[1]]).strip() # get all text until relevant label
    clean_span = re.sub(r'\s{2,}', ' ', clean_span)
    start = span[1] # switch start to end of previous label
    label_text = sentence[span[0]:span[1]]
    l = get_label(label_text, regex, d)
    if l != [] and clean_span != '': # if there is label AND clean span is not empty, append 
      labels.append(l)
      spans_text.append(clean_span)
      clean_idxs = list(re.finditer(re.escape(clean_span), clean_sentence))[0].span()
      start_idxs.append(clean_idxs[0])
      end_idxs.append(clean_idxs[1])

  return labels, spans_text, clean_sentence, start_idxs, end_idxs


def create_missing_idxs(sentence_df):
  # after calculating metrics for each label, replace missing labels with 'NO_ANNOTATION' label
  # also add start = 0 and end = end of sentence
  d = create_label_dict()
  new_labels = []
  new_start_idxs = []
  new_end_idxs = []
  for idx, s in sentence_df['start_idxs'].items():
    e = sentence_df['end_idxs'].iloc[idx]
    l = sentence_df['labels'].iloc[idx]
    sentence = sentence_df['sentences'].iloc[idx]
    if s == []:
      s = [0]
    if e == []:
      e = [len(sentence)]
    if l == []:
      l = [d['NO_ANNOTATION']]
    new_start_idxs.append(s)
    new_end_idxs.append(e)
    new_labels.append(l)
  sentence_df['start_idxs'] = new_start_idxs
  sentence_df['end_idxs'] = new_end_idxs
  sentence_df['labels'] = new_labels
  return sentence_df

def get_sentences_labels(annot_df):
  # wrapper function to preprocess raw labelled notes dataset into sentences with start/end of spans and labels
  d = create_label_dict()
  annot_df = process_annot_df(annot_df)
  sentence_df = pd.DataFrame()
  ids = []
  all_labels = []
  sentences = []
  orig_sentences = []
  all_start_idxs = []
  all_end_idxs = []
  spans = []
  n_spans = []

  for idx, note in annot_df['notes'].items():
    note =  re.sub('\d\.\d', '1', note)
    note = re.sub('Ms\.|Mr\.', '', note)
    sents = note.split('.') # using more sophisticated sentencizer eg Spacy too inconsistent/unpredictable
    for sentence in sents:
      orig_sentences.append(sentence)
      ids.append(idx)
      labels, spans_text, clean_sentence, start_idxs, end_idxs = get_spans(sentence, d)
      # print(labels, start_idxs, end_idxs, spans_text, clean_sentence, repr(sentence))
      all_labels.append(labels)
      sentences.append(clean_sentence)
      all_start_idxs.append(start_idxs)
      all_end_idxs.append(end_idxs)
      spans.append(spans_text)
      n_spans.append(len(labels))
      assert len(labels) == len(start_idxs)

  sentence_df['ids'] = ids
  sentence_df['start_idxs'] = all_start_idxs
  sentence_df['end_idxs'] = all_end_idxs
  sentence_df['labels'] = all_labels
  sentence_df['sentences'] = sentences
  sentence_df['original_sents'] = orig_sentences
  sentence_df['spans'] = spans
  sentence_df['n_spans'] = n_spans
  sentence_df = sentence_df[sentence_df['sentences'].str.len() > 2].reset_index(drop=True)

  sentence_df = clean_df(sentence_df).reset_index(drop=True)
  sentence_df = create_missing_idxs(sentence_df).reset_index(drop=True)
  print(count_labels(sentence_df))
  return sentence_df

def count_labels(sentence_df):
  d = create_label_dict()
  d_labels = {}
  if type(sentence_df['labels'].iloc[0]) != list:
    sentence_df['labels'] = sentence_df['labels'].apply(lambda x: literal_eval(x))
  for l in sentence_df['labels']:
    for i in l:
      if i in d_labels:
        d_labels[i]+=1
      else:
        d_labels[i]=1
  return {[i for i in d if d[i]==k][0]: v for k, v in sorted(d_labels.items(), key=lambda item: item[1], reverse=True)}


def create_merged_labels_dict():
    d = create_label_dict()
    merged_labels = {}
    seen_values = set()
    idx = 0
    for key, value in d.items():
        match = re.search(r'(.+)_(POSITIVE|NEGATIVE)$', key)
        if match:
            base_key = match.group(1)
            if 'POSITIVE' in key:
                pair_value =  d[base_key + '_NEGATIVE']
            else:
                pair_value = d[base_key + '_POSITIVE']
            if value not in seen_values and pair_value not in seen_values:
                seen_values.update([value, pair_value])
                merged_labels[(value, pair_value)] = idx
                idx+=1
        else:
            merged_labels[value] = idx
            idx +=1
   
    return merged_labels


def get_merged_label(l, merged_labels):
    d = create_label_dict()
    no_annot = [d[i] for i in d if i == 'NO_ANNOTATION'][0]
    k = set(list(merged_labels.keys()))
    k.remove(no_annot)
    merged = []
    for i in l:
        if i == no_annot:
            m = merged_labels[no_annot]
        else:
            for key in k:
                if i in key:
                    m = merged_labels[key]
        merged.append(m)
    return merged