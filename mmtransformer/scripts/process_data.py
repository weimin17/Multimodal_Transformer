import sys
sys.path.insert(0, '../mimic3-benchmarks')

#import tensorflow as tf
import numpy as np
import argparse
import os
import imp
import re

from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader

from mimic3models.preprocessing import Discretizer, Normalizer
#from mimic3models import metrics
#from mimic3models import keras_utils
from mimic3models import common_utils

from utility import TextReader, merge_text_raw
from config import Config

args = Config()

# Build readers, discretizers, normalizers
'''reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                   listfile=os.path.join(
    args.data, 'train_listfile.csv'),
    period_length=48.0)'''

reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                   listfile=os.path.join(
                                       args.data, 'val_listfile.csv'),
                                   period_length=48.0)

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(
    reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(
    discretizer_header) if x.find("->") == -1]

# choose here which columns to standardize
normalizer = Normalizer(fields=cont_channels)
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(
        args.timestep, args.imputation)
    normalizer_state = os.path.join(
        os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

normalizer = None
train_raw = utils.load_data(
    reader, discretizer, normalizer, args.small_part, return_names=True)
#val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)

print(len(train_raw['names']))

'''
1. Read text based on patient_id
2. merge text and time series based on patient_id
3. create numpy arrays
4. call Model functions.
5. train and evaluate.
'''

treader = TextReader(args.textdata)
train_text = treader.read_all_text_concat(train_raw['names'])
#val_raw = treader.read_all_text_concat(val_raw[''])
print(len(train_text))

raw = merge_text_raw(train_text, train_raw)

import pickle
with open(args.evalpicklepath, 'wb') as f:
    pickle.dump(raw, f, protocol=pickle.HIGHEST_PROTOCOL)

print(raw[0].shape)
