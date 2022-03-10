import json
import pandas as pd
import os
import config
# import tensorflow as tf
from sklearn import metrics
import sklearn
from sklearn.metrics import accuracy_score
import logging
import pickle
import argparse
import sys
import numpy as np
from matplotlib import pyplot
sys.path.insert(0, '..')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="train/test/eval")
    parser.add_argument("--load_model", help="1/0 to specify whether to load the model", default="0")
    parser.add_argument("--number_epoch", default="10", help="Number of epochs to train the model")
    parser.add_argument("--batch_size", default="5")
    parser.add_argument("--model_type", default='both', help="Which model to use, 'both' uses both the text model and time-series model, 'baseline' only uses time-series model, 'text_only' only uses text model")
    parser.add_argument( "--checkpoint_path", help="Path for checkpointing", default='bs')
    parser.add_argument("--freeze_model", default="0")
    parser.add_argument('--TextModelCheckpoint', help='Checkpoint path for only text model to load')
    parser.add_argument('--TSModel', help='Model to use for time series part of data, the options are: 1. LSTM: LSTM model 2. BiLSTM: Bidirectional LSTM model 3. Transformer: Transformer on Time series.', default='LSTM')
    parser.add_argument('--gpu_id', help='Which gpu to use', default='2', type=str)
    parser.add_argument('--MaxLen', help='maximum length of text to use', type=int, default=128)
    parser.add_argument('--NumOfNotes', help='Number of notes to include for a patient input 0 for all the notes.', type=int, default=5)

    parser.add_argument("--EmbedModel", default="BioBert", help="The model for using as text model when using bert based models options are 'Bert' simple bert which is not pre-trained on clinical notes, 'BioBert' Bio+Clinical notes bert which is pre-trained on clinical notes, 'bioRoberta' roberta model pre-trained on medical papers, 'MedBert' is pre-trained on pub papers.")
    parser.add_argument("--model_name", default='BioBert', help="'BioBert' uses bert based models this achieves the best results")

    parser.add_argument('--Seed', help='Seed to use for both torch and np.random', type=int)
    parser.add_argument('--LR', help='Learning rate to train the model', type=float, default=2e-5)

    parser.add_argument("--log_file")

    args = vars(parser.parse_args())
    assert args['mode'] in ['train', 'test', 'eval']
    # args['decay'] = float(args['decay'])
    return args


def get_config():
    return config.Config()


def get_embedding_dict(conf):
    with open(conf.model_path, 'rb') as f:
        data = pickle.load(f)

    index2word_tensor = data["model"]["index2word"]
    index2word_tensor.pop()
    index2word_tensor.append('<pad>')
    word2index_lookup = {word: index for index,
                         word in enumerate(index2word_tensor)}
    vectors = data["model"]["vectors"]

    return vectors, word2index_lookup


#def get_logger(log_file):
#    # get TF logger
#    log = logging.getLogger('tensorflow')
#    log.setLevel(logging.DEBUG)
#
#    # create formatter and add it to the handlers
#    formatter = logging.Formatter(
#        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
#    # create file handler which logs even debug messages
#    fh = logging.FileHandler(log_file)
#    fh.setLevel(logging.INFO)
#    fh.setFormatter(formatter)
#    log.addHandler(fh)
#    return log


def lookup(w2i_lookup, x):
    if x in w2i_lookup:
        return w2i_lookup[x]
    else:
        return len(w2i_lookup)


class AUCPR():
    def __init__(self, *args, **kwargs):
        self.y_true = None
        self.y_pred = None

    def add(self, pred, y_true):
        if self.y_pred is None:
            self.y_pred = pred
        else:
            self.y_pred = np.concatenate([self.y_pred, pred])

        if self.y_true is None:
            self.y_true = y_true
        else:
            self.y_true = np.concatenate([self.y_true, y_true])

    def get(self):
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            self.y_true, self.y_pred)
        auprc = metrics.auc(recalls, precisions)
        return auprc

    def save(self, name):
        fname = name + ".pkl"
        with open(fname, 'wb') as f:
            pickle.dump((self.y_pred, self.y_true), f, pickle.HIGHEST_PROTOCOL)


def compute_metrics(y_test, y_pred):
    '''
    Get metrics. 
    '''
    import sklearn
    import numpy as np
    from sklearn.metrics import accuracy_score

    # y_test, y_pred should be 1D array
    y_test, y_pred = np.array( y_test ), np.array( y_pred )
    acc= accuracy_score(y_test, y_pred)
    auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
    recall=sklearn.metrics.recall_score(y_test, y_pred)
    precision=sklearn.metrics.precision_score(y_test, y_pred)
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    f1 = sklearn.metrics.f1_score(y_test, y_pred)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])

    print('Label positive {}, Label negative {}, Total {}'.format( np.sum(y_test==1), np.sum(y_test==0), len(y_test)  ) )
    print('acc {:.4f}, auc {:.4f}, recall {:.4f}, precision {:.4f}, F1 {:.4f}, \ncm {}'.format(acc, auc, recall, precision, f1, cm)  )
    
    return acc, auc, recall, precision, f1, cm

class Eval_Metrics():
    def __init__(self, *args, **kwargs):
        self.y_true = None
        self.y_pred = None

    def add(self, pred, y_true):
        if self.y_pred is None:
            self.y_pred = pred
        else:
            self.y_pred = np.concatenate([self.y_pred, pred])

        if self.y_true is None:
            self.y_true = y_true
        else:
            self.y_true = np.concatenate([self.y_true, y_true])

    def get_aucpr(self):
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            self.y_true, self.y_pred)
        auprc = metrics.auc(recalls, precisions)
        return auprc

    def get_acc(self):
        acc = accuracy_score(self.y_true, self.y_pred)
        return acc

    def get_precision(self):
        precision = sklearn.metrics.precision_score(self.y_true, self.y_pred)
        return precision

    def get_recall(self):
        recall = sklearn.metrics.recall_score(self.y_true, self.y_pred)
        return recall

    def get_f1(self):
        f1 = sklearn.metrics.f1_score(self.y_true, self.y_pred)
        return f1

    def get_sensitivity(self):
        cm = sklearn.metrics.confusion_matrix(self.y_true, self.y_pred)
        sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
        return sensitivity

    def get_specificity(self):
        cm = sklearn.metrics.confusion_matrix(self.y_true, self.y_pred)
        specificity = cm[1,1]/(cm[1,0]+cm[1,1])
        return specificity



    def save(self, name):
        fname = name + ".pkl"
        with open(fname, 'wb') as f:
            pickle.dump((self.y_pred, self.y_true), f, pickle.HIGHEST_PROTOCOL)














class MetricPerHour():
    def __init__(self):
        self.y_true_hr = {}
        self.pred_hr = {}
        self.aucpr = {}

        self.y_true = None
        self.y_pred = None

        self.metric_type = 'aucpr'

    def add(self, pred, y_true, mask):
        # pred and y_true are both 3d tensors.
        pred = np.squeeze(pred, axis=-1)
        y_true = np.squeeze(y_true, axis=-1)
        assert len(pred.shape) == 2, "Pred: {} Y: {} Mask:{}".format(
            str(pred.shape), str(y_true.shape), str(mask.shape))
        assert len(y_true.shape) == 2
        assert len(mask.shape) == 2
        for hour in range(5, y_true.shape[1]):
            y_true_h = y_true[:, hour]
            pred_h = pred[:, hour]
            mask_h = mask[:, hour]
            mask_h = np.squeeze(mask_h.astype(np.bool))
            # Fix this
            y_true_h = y_true_h[mask_h]
            pred_h = pred_h[mask_h]

            if len(mask_h.shape) == 0:
                # print("Failed: Mask: {} y_pred: {} y_true: {}".format(
                #    str(mask_h.shape), str(pred_h.shape), str(y_true_h.shape)))
                continue

            if hour not in self.y_true_hr:
                self.y_true_hr[hour] = y_true_h
                self.pred_hr[hour] = pred_h
            else:
                self.y_true_hr[hour] = np.concatenate(
                    [self.y_true_hr[hour], y_true_h])
                self.pred_hr[hour] = np.concatenate(
                    [self.pred_hr[hour], pred_h])

            if self.y_true is None:
                self.y_true = y_true_h
                self.y_pred = pred_h
            else:
                self.y_true = np.concatenate([self.y_true, y_true_h])
                self.y_pred = np.concatenate([self.y_pred, pred_h])

    def get_metric(self, y_true, pred):
        if self.metric_type == 'aucpr':
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(
                y_true, pred)
            value = metrics.auc(recalls, precisions)
        elif self.metric_type == 'kappa':
            value = metrics.cohen_kappa_score(y_true, pred, weights='linear')
        return value

    def get(self):
        self.aucpr = {}
        for hour in self.y_true_hr.keys():
            y_true = self.y_true_hr[hour]
            pred = self.pred_hr[hour]
            try:
                self.aucpr[hour] = self.get_metric(y_true, pred)
            except:
                print("Failed get() for hour: {},Y_true: {}, Pred: {}".format(
                    hour, str(y_true.shape), str(pred.shape)))
        aggregated = self.get_metric(self.y_true, self.y_pred)
        return self.y_true_hr, self.pred_hr, self.aucpr, aggregated

    def save(self, name):
        fname = name + ".pkl"
        with open(fname, 'wb') as f:
            pickle.dump({'aucpr': self.aucpr, 'predbyhr': self.pred_hr, 'truebyhr': self.y_true_hr},
                        f, pickle.HIGHEST_PROTOCOL)


class AUCPRperHour(MetricPerHour):
    def __init__(self):
        super().__init__()
        self.metric_type = 'aucpr'


class KappaPerHour(MetricPerHour):
    def __init__(self):
        super().__init__()
        self.metric_type = 'kappa'


def pplloott(cnn_p, cnn_r, baseline_p, baseline_r):
    pyplot.close()
    pyplot.plot(cnn_p, cnn_r, marker='.', linestyle='dashed',
                color='red', linewidth=1, markersize=1, label='MultiModal')
    pyplot.plot(baseline_p, baseline_r, marker='.', linestyle='dashed',
                color='green', linewidth=1, markersize=1, label='Baseline')
    pyplot.title('In-Hospital Mortality')
    pyplot.legend(('MultiModal', 'Baseline'), loc='upper right')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.savefig('PR.png')


def timeplot(baseline_aucpr, text_aucpr, true_values, HOUR=3*24*7,
             cummulative=False, baseline=None, text=None):
    from matplotlib import pyplot
    import math
    hours = []
    baseline_values = []
    text_values = []

    number_of_episodes_with_mortal_label = []
    number_of_episodes = []

    for hour in sorted(baseline_aucpr.keys())[:HOUR]:
        if not (math.isnan(baseline_aucpr[hour]) or math.isnan(text_aucpr[hour])):
            baseline_values.append(baseline_aucpr[hour])
            text_values.append(text_aucpr[hour])
            hours.append(hour)
        number_of_episodes_with_mortal_label.append(true_values[hour].sum())
        number_of_episodes.append(true_values[hour].shape[0])

    print("Len of hours", len(hours))
    pyplot.close()
    # pyplot.plot(hours, baseline_values, '.', color='green')
    # pyplot.plot(hours, text_values, '.', color='red')
    pyplot.plot(hours, baseline_values, marker='.', linestyle='dashed',
                color='green', linewidth=1, markersize=1, label='Baseline')
    pyplot.plot(hours, text_values, marker='.', linestyle='dashed',
                color='red', linewidth=1, markersize=1, label='MultiModal')
    pyplot.title('Decompensation')
    pyplot.legend(('Baseline', 'MultiModal'), loc='upper right')
    pyplot.ylabel('AUCPR')
    pyplot.xlabel('HOURs from admission')
    pyplot.savefig('AUCPR_decom.png')

    pyplot.close()
    pyplot.plot(sorted(baseline_aucpr.keys())[
                :HOUR], number_of_episodes_with_mortal_label)
    pyplot.title('Total patients with mortality=1 in testset')
    pyplot.ylabel('Count')
    pyplot.xlabel('HOURs from admission')
    pyplot.savefig('Data1.png')

    pyplot.close()
    pyplot.plot(sorted(baseline_aucpr.keys())[
                :HOUR], number_of_episodes)
    pyplot.title('Total patients in testset')
    pyplot.ylabel('Count')
    pyplot.xlabel('HOURs from admission')
    pyplot.savefig('Data2.png')

    pyplot.close()
    pyplot.plot(sorted(baseline_aucpr.keys())[
                :HOUR], np.array(number_of_episodes_with_mortal_label) /
                np.array(number_of_episodes))
    pyplot.title('Ratio of patient with mortality=1 at that time.')
    pyplot.ylabel('fraction')
    pyplot.xlabel('HOURs from admission')
    pyplot.savefig('Data3.png')

    if cummulative:
        tyb = np.array([])
        pvb = np.array([])
        tyc = np.array([])
        pvc = np.array([])
        hours = []
        baseline_values = []
        text_values = []

        for hour in sorted(baseline_aucpr.keys())[:HOUR]:
            pvb = np.concatenate([pvb, baseline['predbyhr'][hour]])
            tyb = np.concatenate([tyb, baseline['truebyhr'][hour]])
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(tyb, pvb)
            aucpr = metrics.auc(recalls, precisions)
            hours.append(hour)
            baseline_values.append(aucpr)

            pvc = np.concatenate([pvc, text['predbyhr'][hour]])
            tyc = np.concatenate([tyc, text['truebyhr'][hour]])
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(tyc, pvc)
            aucpr = metrics.auc(recalls, precisions)
            text_values.append(aucpr)

        pyplot.close()
        pyplot.plot(hours, baseline_values, marker='.', linestyle='dashed',
                    color='green', linewidth=1, markersize=1, label='Baseline')
        pyplot.plot(hours, text_values, marker='.', linestyle='dashed',
                    color='red', linewidth=1, markersize=1, label='MultiModal')
        pyplot.title('Decompensation')
        pyplot.legend(('Baseline', 'MultiModal'), loc='upper right')
        pyplot.ylabel('AUCPR - cumulative')
        pyplot.xlabel('HOURs from admission')
        pyplot.savefig('AUCPR_decom_cum.png')


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def get_pos_enc(num_units, T):
    item = tf.constant([1.0 / np.power(10000, 2.*i/num_units)
                        for i in range(num_units)])
    item = tf.reshape(item, (1, num_units))
    itemR = tf.tile(item, [T, 1])
    pos = tf.to_float(tf.range(T))
    pos = tf.reshape(pos, (T, 1))
    a = pos * itemR
    e = tf.sin(a[:, 0::2])  # dim 2i
    o = tf.cos(a[:, 1::2])  # dim 2i+1
    y = tf.reshape(tf.stack([e, o], axis=1), [tf.shape(e)[0], -1])
    return y


def positional_encoding(inputs, T, N,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.
    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        '''position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])'''
        # position_enc = get_pos_enc(num_units, T)

        # Second part, apply the cosine to even columns and sin to odds.
        # position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        # position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        # lookup_table = tf.convert_to_tensor(position_enc)
        lookup_table = get_pos_enc(num_units, T)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

    return outputs


def multihead_attention(queries,
                        keys,
                        dropout_rate,
                        num_units=None,
                        num_heads=10,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units,
                            activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(
            keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(
            keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2),
                       axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2),
                       axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2),
                       axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [
            1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings,
                           outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(
                diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [
                tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings,
                               outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(
            tf.abs(queries), axis=-1))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(
            query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.nn.dropout(
            outputs, keep_prob=dropout_rate)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads,
                                     axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def feedforward(inputs,
                num_units=[1200, 300],
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs


def generate_tensor_text(t, w2i_lookup, conf):
    t_new = []
    max_len = -1
    for text in t:
        tokens = list(map(lambda x: lookup(w2i_lookup, x), str(text).split()))
        if conf.max_len > 0:
            tokens = tokens[:conf.max_len]
        t_new.append(tokens)
        max_len = max(max_len, len(tokens))
    pad_token = w2i_lookup['<pad>']
    for i in range(len(t_new)):
        if len(t_new[i]) < max_len:
            t_new[i] += [pad_token] * (max_len - len(t_new[i]))
    return np.array(t_new)


class TextInformationReader():
    def _get_time_hours(self, text_data):
        text_timestamps = text_data[0]
        text = text_data[1]
        text_timestamps = np.array(text_timestamps, dtype=np.datetime64)
        start_time = text_timestamps[0]
        text_timestamps = text_timestamps - start_time
        text_timestamps = text_timestamps.astype(
            "timedelta64[h]").astype(np.int32)
        return (text_timestamps, text)

    def build_text_dictionary(self, data):
        self.text_lookup_table = {}
        for name, text_ts_data in zip(data[3], data[4]):
            self.text_lookup_table[name] = self._get_time_hours(text_ts_data)
        return self.text_lookup_table

    def __init__(self, data):
        self.build_text_dictionary(data)

    def get_text_till_hour(self, name, hour):
        if name not in self.text_lookup_table:
            return ""
        row = self.text_lookup_table[name]
        # print(name, hour, max(row[0]))
        # assert hour <= max(row[0]), 'Max hour check failed'
        t = ""
        current_index = 0
        while current_index < len(row[0]) and row[0][current_index] <= hour:
            t += str(row[1][current_index])
            current_index += 1
        return t

    def get_text_till_hours(self, names, hours):
        texts = []
        for (name, hour) in zip(names, hours):
            texts.append(self.get_text_till_hour(name, hour))
        return texts


def diff(time1, time2):
    # compute time2-time1
    # return difference in hours
    a = np.datetime64(time1)
    b = np.datetime64(time2)
    h = (b-a).astype('timedelta64[h]').astype(int)
    '''if h < -1e-6:
        print(h)
        assert h > 1e-6'''
    return h

def diff_float(time1, time2):
    # compute time2-time1
    # return differences in hours but as float
    h = (time2-time1).astype('timedelta64[m]').astype(int)
    return h/60.0

class TextReader():
    def __init__(self, dbpath, starttime_path):
        self.dbpath = dbpath
        self.all_files = set(os.listdir(dbpath))
        with open(starttime_path, 'rb') as f:
            self.episodeToStartTime = pickle.load(f)

    def get_name_from_filename(self, fname):
        # '24610_episode1_timeseries.csv'
        tokens = fname.split('_')
        pid = tokens[0]
        episode_id = tokens[1].replace('episode', '').strip()
        return pid, episode_id

    def read_all_text_events(self, names):
        texts = {}
        for name in names:
            pid, eid = self.get_name_from_filename(name)
            text_file_name = pid + "_" + eid
            if text_file_name in self.all_files:
                texts[name] = self.read_text_event_json(text_file_name)
        # for each filename (which contains pateintid, eid) and can be used to merge.
        # it will store a list with timestep and text at that time step.
        return texts
    
    def read_all_text_concat_json(self, names, period_length=48.0):
        texts_dict = {}
        time_dict = {}
        start_times = {}
        for patient_id in names:
            pid, eid = self.get_name_from_filename(patient_id)
            text_file_name = pid + "_" + eid
            if text_file_name in self.all_files:
                time, texts = self.read_text_event_json(text_file_name)
                start_time = self.episodeToStartTime[text_file_name]
                if len(texts) == 0 or start_time == -1:
                    continue
                final_concatenated_text = ""
                times_array = []
                for (t, txt) in zip(time, texts):
                    if diff(start_time, t) <= period_length + 1e-6:
                        final_concatenated_text = final_concatenated_text + " "+txt
                        times_array.append(t)
                    else:
                        break
            texts_dict[patient_id] = final_concatenated_text
            time_dict[patient_id] = times_array
            start_times[patient_id] = start_time
        return texts_dict, time_dict, start_times
    
    def read_all_text_append_json(self, names, period_length=48.0, NumOfNotes=5, notes_aggeregate='Mean'):
        texts_dict = {}
        time_dict = {}
        start_times = {}
        for patient_id in names:
            pid, eid = self.get_name_from_filename(patient_id)
            text_file_name = pid + "_" + eid
            if text_file_name in self.all_files:
                time, texts = self.read_text_event_json(text_file_name)
                start_time = self.episodeToStartTime[text_file_name]
                if len(texts) == 0 or start_time == -1:
                    continue
                final_concatenated_text = []
                times_array = []
                for (t, txt) in zip(time, texts):
                    if diff(start_time, t) <= period_length + 1e-6:
                        final_concatenated_text.append(txt)
                        times_array.append(t)
                    else:
                        break
                # if notes_aggeregate == 'First':
                #     texts_dict[patient_id] = final_concatenated_text[:NumOfNotes]
                #     time_dict[patient_id] = times_array[:NumOfNotes]
                # else:
                    # texts_dict[patient_id] = final_concatenated_text[-NumOfNotes:]
                    # time_dict[patient_id] = times_array[-NumOfNotes:]
                texts_dict[patient_id] = final_concatenated_text
                time_dict[patient_id] = times_array
                start_times[patient_id] = start_time
        
        return texts_dict, time_dict, start_times

    def read_text_event_json(self, text_file_name):
        filepath = os.path.join(self.dbpath, str(text_file_name))
        with open(filepath, 'r') as f:
            d = json.load(f)
        time = sorted(d.keys())
        text = []
        for t in time:
            text.append(" ".join(d[t]))
        assert len(time) == len(text)
        return time, text


def merge_text_raw(textdict, raw, timedict, start_times):
    mask = []
    text = []
    names = []
    times = []
    start_times_arr = []
    suceed = 0
    missing = 0
    for item in raw['names']:
        if item in textdict:
            mask.append(True)
            text.append(textdict[item])
            times.append(timedict[item])
            start_times_arr.append(start_times[item])
            names.append(item)
            suceed += 1
        else:
            mask.append(False)
            missing += 1

    print("Suceed Merging: ", suceed)
    print("Missing Merging: ", missing)

    data = [[], [], [], [], [], []]
    data[0] = raw['data'][0][mask]
    data[1] = np.array(raw['data'][1])[mask]
    data[2] = text
    data[3] = names
    data[4] = times
    data[5] = start_times_arr
    # X,y,T,names
    return data

def get_text_sep(textdict, raw, timedict, start_times):
    text = []
    X_data = []
    Ys = []
    names = []
    times = []
    start_times_arr = []
    suceed = 0
    missing = 0
    for indx, item in enumerate(raw['names']):
        if item in textdict:
            for i, txt in enumerate(textdict[item]):
                X_data.append(raw['data'][0][indx])
                Ys.append(raw['data'][1][indx])
                text.append(txt)
                times.append(timedict[item][i])
                start_times_arr.append(start_times[item])
                names.append(item)
                suceed += 1
        else:
            missing += 1

    print("Suceed Merging: ", suceed)
    print("Missing Merging: ", missing)

    data = [[], [], [], [], [], []]
    data[0] = X_data
    data[1] = Ys
    data[2] = text
    data[3] = names
    data[4] = times
    data[5] = start_times_arr
    # X,y,T,names
    return data

def train_data_plot(data):
    hour_map_mortality_count = {}
    hour_map_population_count = {}
    HOUR = 3*24*7
    for batch in data:
        mask = batch['Mask']
        y_true = batch['Output']
        for hour in range(5, mask.shape[1]):
            y_true_h = y_true[:, hour]
            if hour in hour_map_mortality_count:
                hour_map_mortality_count[hour] += y_true_h.sum()
                hour_map_population_count[hour] += mask[:, hour].sum()
            else:
                hour_map_mortality_count[hour] = y_true_h.sum()
                hour_map_population_count[hour] = mask[:, hour].sum()

    hours = sorted(hour_map_mortality_count.keys())[:HOUR]
    mortal = []
    total = []
    for h in hours:
        mortal.append(hour_map_mortality_count[h])
        total.append(hour_map_population_count[h])

    pyplot.close()
    pyplot.plot(hours, mortal)
    pyplot.title('Total patients with mortality=1 in train set')
    pyplot.ylabel('Count')
    pyplot.xlabel('HOURs from admission')
    pyplot.savefig('Train_Data1.png')

    pyplot.close()
    pyplot.plot(hours, total)
    pyplot.title('Total patients in trainset')
    pyplot.ylabel('Count')
    pyplot.xlabel('HOURs from admission')
    pyplot.savefig('Train_Data2.png')

    pyplot.close()
    pyplot.plot(hours, np.array(mortal) / np.array(total))
    pyplot.title(
        'Ratio of patient with mortality=1 with total number of patient against time.')
    pyplot.ylabel('fraction')
    pyplot.xlabel('HOURs from admission')
    pyplot.savefig('Train_Data3.png')
