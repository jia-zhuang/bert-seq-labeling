
import collections
import csv
import os
import re
import numpy as np
import modeling
import optimization
from tokenizers import BertWordPieceTokenizer
import tensorflow as tf
from seqeval import metrics
from utils import PextProcessor, file_based_convert_examples_to_features

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")


def file_based_input_fn_builder(input_file, seq_length, is_training, batch_size):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        return example

    def input_fn():
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size)
            )

        return d

    return input_fn


def create_model(bert_config, num_labels, is_training, input_ids, input_mask, segment_ids):
    """Creates a sequence labeling model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids)

    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    # batch_size = final_hidden_shape[0]
    # seq_length = final_hidden_shape[1]
    # hidden_size = final_hidden_shape[2]

    classifier = tf.layers.Dense(
            num_labels, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="classifier"
    )

    # loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none", from_logits=True)

    if is_training:
        final_hidden = tf.nn.dropout(final_hidden, keep_prob=0.9)
    
    logits = classifier(final_hidden)  # B x L x 3

    return logits

    # outputs = (logits,)
    # if labels is not None:
    #     logits = tf.reshape(logits, (-1, num_labels))  # B*L x H
    #     active_loss = tf.reshape(input_mask, (-1,))  # B*L
    #     active_logits = tf.boolean_mask(logits, active_loss)
    #     labels = tf.reshape(labels, (-1,))  # B*L
    #     active_labels = tf.boolean_mask(labels, active_loss)
    #     cross_entropy = loss_fct(active_labels, active_logits)
    #     loss = tf.reduce_sum(cross_entropy) / tf.to_float(batch_size)
    #     outputs = (loss,) + outputs

    # return outputs


def model_fn(features, labels, mode, params):
    """
        features: input_ids, input_mask, segment_ids, label_ids
        labels: we don't use it, just None
        mode: TRAIN, EVAL, PREDICT
        params: {'bert_config': None, 'num_labels': None, 
                 'init_checkpoint': None, 'learning_rate': None,
                 'num_train_steps': None, 'num_warmup_steps': None}
    """
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
        tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    logits = create_model(params['bert_config'], params['num_labels'], is_training, 
                          input_ids, input_mask, segment_ids)

    # load weights from checkpoint
    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names
    ) = modeling.get_assignment_map_from_checkpoint(tvars, params['init_checkpoint'])
    
    tf.train.init_from_checkpoint(params['init_checkpoint'], assignment_map)
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        labels = features["label_ids"]
        batch_size = tf.shape(labels)[0]  # don't use labels.shape[0]
        loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        logits = tf.reshape(logits, (-1, params['num_labels']))  # B*L x H
        active_loss = tf.reshape(input_mask, (-1,))  # B*L
        active_logits = tf.boolean_mask(logits, active_loss)
        labels = tf.reshape(labels, (-1,))  # B*L
        active_labels = tf.boolean_mask(labels, active_loss)
        cross_entropy = loss_fct(active_labels, active_logits)
        loss = cross_entropy
        #loss = tf.reduce_sum(cross_entropy) / tf.to_float(batch_size)

        train_op = optimization.create_optimizer(
            loss, params['learning_rate'], params['num_train_steps'],
            params['num_warmup_steps'], use_tpu=False)
        
        output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.PREDICT:
        pred_ids = tf.argmax(logits, axis=-1)
        label_ids = features['label_ids']
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'pred_ids': pred_ids, 'label_ids': label_ids}
        )

    else:
        output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions={"logits": logits},)
    
    return output_spec


def get_last_ckpt_prefix(output_dir):
    PAT = re.compile(r'model.ckpt-(\d+)\.index')
    max_steps = 0
    for file in os.listdir(output_dir):
        match = PAT.match(file)
        if match:
            steps = int(match.group(1))
            max_steps = max(max_steps, steps)

    return os.path.join(output_dir, f'model.ckpt-{max_steps}')


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    processor = PextProcessor()
    label_list = processor.get_labels()
    # load tokenizer
    tokenizer = BertWordPieceTokenizer(FLAGS.vocab_file, add_special_tokens=False)
    tokenizer.no_padding()
    tokenizer.no_truncation()
    
    # Create Estimator
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    
    params = {
        "bert_config": bert_config,
        "num_labels": len(label_list),
        "init_checkpoint": FLAGS.init_checkpoint,
        "learning_rate": FLAGS.learning_rate,
        "num_train_steps": num_train_steps,
        "num_warmup_steps": num_warmup_steps
    }
    
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.output_dir,
        params=params
    )

    if FLAGS.do_train:
        tf.gfile.MakeDirs(FLAGS.output_dir)
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            batch_size=FLAGS.train_batch_size)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        # save config
        out_config_file = os.path.join(FLAGS.output_dir, 'config.json')
        with tf.gfile.GFile(out_config_file, "w") as writer:
            writer.write(bert_config.to_json_string())
        
        # save tokenizer
        tokenizer.save(FLAGS.output_dir)
    
    if FLAGS.do_eval:
        # load last ckpt
        ckpt_prefix = get_last_ckpt_prefix(FLAGS.output_dir)
        params = {
            'bert_config': modeling.BertConfig.from_json_file(os.path.join(FLAGS.output_dir, 'config.json')),
            'num_labels': len(label_list),
            'init_checkpoint': get_last_ckpt_prefix(FLAGS.output_dir),
        }
        estimator = tf.estimator.Estimator(model_fn=model_fn, params=params)

        # prepare data
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        #file_based_convert_examples_to_features(
        #    eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
        
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            batch_size=FLAGS.eval_batch_size)
        
        result = estimator.predict(input_fn=eval_input_fn, yield_single_examples=False)
        
        pred_ids, label_ids = [], []
        for prediction in result:
            pred_ids.append(prediction['pred_ids'])
            label_ids.append(prediction['label_ids'])
        
        pred_ids = np.concatenate(pred_ids, axis=0)
        label_ids = np.concatenate(label_ids, axis=0)

        y_pred = [[] for _ in range(label_ids.shape[0])]
        y_true = [[] for _ in range(label_ids.shape[0])]
        
        pad_token_label_id = 0
        for i in range(label_ids.shape[0]):
            for j in range(label_ids.shape[1]):
                if label_ids[i, j] != pad_token_label_id:
                    y_pred[i].append(label_list[pred_ids[i, j]])
                    y_true[i].append(label_list[label_ids[i, j]])

        report = metrics.classification_report(y_true, y_pred, digits=4)
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)

        tf.logging.info("Eval result: \n" + report)

        out_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(out_eval_file, "w") as writer:
            writer.write(f'precision = {precision: f} \n')
            writer.write(f'recall = {recall: f} \n')
            writer.write(f'f1 = {f1 :f} \n')

    if FLAGS.do_predict:
        pass

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
