import json
from typing import Any
from functools import partial

import datasets
from transformers import PreTrainedTokenizerBase
from nltk import sent_tokenize
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

dataset = datasets.load_dataset('cnn_dailymail', name='3.0.0')

src_text_column_name, tgt_text_column_name = "article", "highlights"
max_source_length, max_target_length = 1024, 128
n_proc = 40

# Since bos_token is used as the beginning of target sequence,
# we use mask_token to represent the beginning of each sentence.
bosent_token = "<mask>"
bosent_token_id = 50264

rouge_scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)


def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        padding: str,
        max_source_length: int,
        max_target_length: int,
        src_text_column_name: str,
        tgt_text_column_name: str,
):
    inputs, targets = [], []
    all_sent_rouge_scores = []
    for i in range(len(examples[src_text_column_name])):
        if examples[src_text_column_name][i] is not None and examples[tgt_text_column_name][i] is not None:
            input_sentences = sent_tokenize(examples[src_text_column_name][i])
            target_sentences = examples[tgt_text_column_name][i].strip()
            rouge_scores = []
            for sent in input_sentences:
                rouge_scores.append(rouge_scorer.score(target_sentences, sent)['rougeLsum'].fmeasure)
            # todo: add bos_token this way is unsafe
            inputs.append(bosent_token.join(input_sentences))
            targets.append(target_sentences.replace('\n', ' ').replace('  ', ' '))
            all_sent_rouge_scores.append(rouge_scores)
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # replace bos_token_id at the begining of document with bosent_token_id
    for i in range(len(model_inputs['input_ids'])):
        model_inputs['input_ids'][i][0] = bosent_token_id

    all_token_sent_id = []
    for sent_tokens in model_inputs['input_ids']:
        sid = -1
        token_sent_id = []
        for tid in sent_tokens:
            if tid == bosent_token_id:
                sid += 1
            if tid == tokenizer.eos_token_id or tid == tokenizer.pad_token_id:
                sid = -1
            token_sent_id.append(sid)
        all_token_sent_id.append(token_sent_id)

    all_token_info_dist = []
    all_sent_bos_idx = []
    for token_sent_id, sent_rouge_scores in zip(all_token_sent_id, all_sent_rouge_scores):
        sent_rouge_scores = sent_rouge_scores[:max(token_sent_id) + 1]  # truncation
        sent_bos_idx = []
        token_info_dist = []
        bos_idx = 0
        for sid in range(max(token_sent_id) + 1):
            tnum = token_sent_id.count(sid)
            sent_score = sent_rouge_scores[sid]
            token_info_dist.extend([sent_score for _ in range(tnum)])
            sent_bos_idx.extend([bos_idx for _ in range(tnum)])
            bos_idx += tnum
        token_info_dist.extend([-1 for _ in range(token_sent_id.count(-1))])
        all_token_info_dist.append(token_info_dist)
        sent_bos_idx.extend([0 for _ in range(token_sent_id.count(-1))])
        all_sent_bos_idx.append(sent_bos_idx)

    for i in range(len(all_token_sent_id)):
        for j in range(len(all_token_sent_id[i])):
            all_token_sent_id[i][j] += 1

    model_inputs['info_distribution'] = all_token_info_dist
    model_inputs['sentence_bos_index'] = all_sent_bos_idx
    model_inputs['sent_id'] = all_token_sent_id

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", use_fast=False)

convert_to_features = partial(
    convert_to_features,
    tokenizer=tokenizer,
    padding='max_length',
    max_source_length=max_source_length,
    max_target_length=max_target_length,
    src_text_column_name=src_text_column_name,
    tgt_text_column_name=tgt_text_column_name,
)
dataset = dataset.map(
    convert_to_features,
    batched=True,
    num_proc=n_proc,
)

cols_to_keep = ["input_ids", "attention_mask", "labels", "info_distribution", "sentence_bos_index", "sent_id"]
dataset.set_format(columns=cols_to_keep)

for split in ['train', 'validation', 'test']:
    with open(f'data/{split}.json', 'w') as outfile:
        for i, example in enumerate(dataset[split]):
            json_string = json.dumps(example)
            outfile.write(json_string + '\n')
