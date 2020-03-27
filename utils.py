import logging
import os
import pandas as pd
import collections
import sys
import tensorflow as tf


logger = logging.getLogger(__name__)


InputExample = collections.namedtuple('InputExample', 'sentence, subject, predicates, object')
InputFeatures = collections.namedtuple('InputFeatures', 'input_ids, input_mask, segment_ids, label_ids, is_real_example')


class PextProcessor:

    def get_train_examples(self, data_dir):
        return self._create_examples(data_dir, 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(data_dir, 'dev')

    def get_test_examples(self, data_dir):
        pass

    def get_labels(self):
        return ['O', 'B-P', 'I-P']

    def _create_examples(self, data_dir, mode):
        file_path = os.path.join(data_dir, '{}.pkl'.format(mode))
        df = pd.read_pickle(file_path)

        examples = []
        for _, row in df.iterrows():
            examples.append(InputExample(
                sentence=row.get('sentence'),
                subject=row.get('subject'),
                predicates=row.get('predicates'),
                object=row.get('object'),
            ))

        return examples

    
def span_contains(outers, insider):
    '''insider区间是否被包含在多个outers区间中的一个
       outer: list of tuple
       insider: tuple
    '''
    assert insider[0] < insider[1]
    for outer in outers:
        assert outer[0] < outer[1]
        if outer[0] <= insider[0] and outer[1] >= insider[1]:
            return True
    return False


def tokenize_predicates(tokenizer, sentence, predicates, max_len, b_label='B-P', i_label='I-P', o_label='O', pad_label='X'):
    # locate predicates in sentence
    predicates_spans = []
    for p in predicates:
        start_idx = sentence.find(p)
        assert start_idx >= 0, f'`{p}(predicate)` not in `{sentence}(sentence)`'
        end_idx = start_idx + len(p)
        predicates_spans.append((start_idx, end_idx))

    # tokenize
    tokened = tokenizer.encode(sentence)
    labels = []
    for token, offset in zip(tokened.tokens, tokened.offsets):
        token_offset = tokened.original_str.offsets(offset)
        if span_contains(predicates_spans, token_offset):
            if len(labels) == 0 or labels[-1] == o_label:
                labels.append(b_label)
            else:
                # if token.startswith('##'):
                # labels.append(pad_label)
                # else:
                labels.append(i_label)
        else:
            labels.append(o_label)

    # use max_len for truncat
    return tokened.ids[:max_len], labels[:max_len], tokened.tokens[:max_len]


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,
                            cls_token='[CLS]', sep_token='[SEP]', pad_token=0):
    
    label_map = {label: i for i, label in enumerate(label_list)}
    tokened = tokenizer.encode(example.subject + ' ' + example.object)
    sent_input_ids, sent_labels, sent_tokens = tokenize_predicates(tokenizer, example.sentence,
                                                                          example.predicates, max_seq_length - len(tokened) - 3)
    # concat
    tokens = [cls_token] + tokened.tokens + [sep_token] + sent_tokens + [sep_token]
    input_ids = [101] + tokened.ids + [102] + sent_input_ids + [102]
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(tokened) + 2) + [1] * (len(sent_input_ids) + 1)
    label_ids = [0] * (len(tokened) + 2) + [label_map[l] for l in sent_labels] + [0]

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids += [pad_token] * padding_length
    input_mask += [pad_token] * padding_length
    segment_ids += [pad_token] * padding_length
    label_ids += [pad_token] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index < 5:
        logger.info("*** Example ***")
        logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

    features = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids, is_real_example=1)

    return features


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
    
        feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)
    
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()