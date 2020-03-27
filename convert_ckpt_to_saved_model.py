import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import logging
import tensorflow as tf
from predicate_extraction import model_fn
from modeling import BertConfig
from utils import PextProcessor


logger = logging.getLogger(__name__)


def get_serving_input_receiver_fn(seq_length = 256):
    '''这种格式的 saved model 在 tf serving 时需要传入 tf.train.Example '''
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    return tf.estimator.export.build_parsing_serving_input_receiver_fn(name_to_features)


def get_raw_serving_input_receiver_fn(seq_length = 256):
    '''这种格式的 saved model 在 tf serving 时只需用 {'inputs': {'f1': [], 'f2': []}} '''
    name_to_features = {
        "input_ids": tf.placeholder(tf.int64, shape=[None, seq_length]),
        "input_mask": tf.placeholder(tf.int64, shape=[None, seq_length]),
        "segment_ids": tf.placeholder(tf.int64, shape=[None, seq_length]),
    }

    return tf.estimator.export.build_raw_serving_input_receiver_fn(name_to_features)


def convert(saved_model_dir, ckpt_prefix, num_labels):
    ckpt_dir = os.path.dirname(ckpt_prefix)
    bert_config = BertConfig.from_json_file(os.path.join(ckpt_dir, 'config.json'))

    params = {
        "bert_config": bert_config,
        "num_labels": num_labels,
        "init_checkpoint": ckpt_prefix,
        "learning_rate": None,
        "num_train_steps": None,
        "num_warmup_steps": None
    }

    estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir=ckpt_dir)

    serving_input_fn = get_raw_serving_input_receiver_fn()

    res = estimator.export_saved_model(saved_model_dir, serving_input_fn)

    logger.info("Export saved model to %s", res.decode('utf8'))


if __name__ == '__main__':
    processor = PextProcessor()
    label_list = processor.get_labels()
    ckpt_prefix = 'models/magi-finetune-annote-finetune/model.ckpt-14629'
    saved_model_dir = './magi_saved_model'
    convert(saved_model_dir, ckpt_prefix, len(label_list))