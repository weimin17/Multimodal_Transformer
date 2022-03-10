# Main file for running the code
'''
Inference on In-hospital Mortality Prediction, with only text model, Fine-tuned Bio+Clinical BERT (MBERT).

'''
import sys
sys.path.append('..')
from config import Config
sys.path.append('../../mimic3-benchmarks/')
from mimic3models import common_utils
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.in_hospital_mortality import utils as ihm_utils


import utils
import Models
import pickle
import numpy as np
import os
import argparse

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

import warnings
import time


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', help='Which gpu to use', default='2', type=str)

args = vars(parser.parse_args())

start_time = time.time()
# Ignoring warnings
warnings.filterwarnings('ignore')

conf = utils.get_config() # Get configs

os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_id']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

## Loading pre-trained model based on EmbedModel argument
EmbedModelName = "BioBert"
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
BioBert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

BioBert = BioBert.to(device)
BioBertConfig = BioBert.config

TextModelCheckpoint = 'BioClinicalBERT_FT'
batch_size = 5
NumOfNotes=10
output_file_name = 'analysis'
MaxLen=128

# Use text model checkpoint to update the bert model to fine-tuned weights
FTmodel = torch.load(os.path.join('Checkpoints', 'BioClinicalBERT_FT')) 
FTmodel.eval()


from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score
def Evaluate(Labels, Preds, PredScores):
    # Get the evaluation metrics like AUC, percision and etc.
    precision, recall, fscore, support = precision_recall_fscore_support(Labels, Preds, average='binary')
    _, _, fscore_weighted, _ = precision_recall_fscore_support(Labels, Preds, average='weighted')
    accuracy = accuracy_score(Labels, Preds)
    confmat = confusion_matrix(Labels, Preds)
    sensitivity = confmat[0,0]/(confmat[0,0]+confmat[0,1])
    specificity = confmat[1,1]/(confmat[1,0]+confmat[1,1])
    roc_macro, roc_micro, roc_weighted = roc_auc_score(Labels, PredScores, average='macro'), roc_auc_score(Labels, PredScores, average='micro'), roc_auc_score(Labels, PredScores, average='weighted')
    prf_test = {'precision': precision, 'recall': recall, 'fscore': fscore, 'fscore_weighted': fscore_weighted, 'accuracy': accuracy, 'confusionMatrix': confmat, 'sensitivity': sensitivity, 'specificity': specificity, 'roc_macro': roc_macro, 'roc_micro': roc_micro, 'roc_weighted': roc_weighted}
    return prf_test


def Evaluate_Model(model, batch, names):
    # Test the model
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        FirstTime = False
        
        eval_obj = utils.Eval_Metrics()
        
        PredScores = None
        for idx, sample in enumerate(batch):
            X = torch.tensor(sample[0], dtype=torch.float).to(device)
            y = torch.tensor(sample[1], dtype=torch.float).to(device)
            text = [torch.tensor(x, dtype=torch.long).to(device) for x in sample[2]]
            attn = [torch.tensor(x, dtype=torch.long).to(device) for x in sample[3]]
            times = [torch.tensor(x, dtype=torch.long).to(device) for x in sample[4]]

            if sample[2].shape[0] == 0:
                continue
            # Run the model
            Logits, Probs = model(text, attn, times)

            Lambd = torch.tensor(0.01).to(device)
            l2_reg = model.get_l2()
            loss = model.criterion(Logits, y)
            loss += Lambd * l2_reg
            epoch_loss += loss.item() * y.size(0)
            predicted = Probs.data > 0.5
            if not FirstTime:
                PredScores = Probs
                TrueLabels = y
                PredLabels = predicted
                FirstTime = True
            else:
                PredScores = torch.cat([PredScores, Probs])
                TrueLabels = torch.cat([TrueLabels, y])
                PredLabels = torch.cat([PredLabels, predicted])
            eval_obj.add(Probs.detach().cpu(), y.detach().cpu())
            prf_test = Evaluate(TrueLabels.detach().cpu(), PredLabels.detach().cpu(), PredScores.detach().cpu())
        prf_test['epoch_loss'] = epoch_loss / TrueLabels.shape[0]
        prf_test['aucpr'] = eval_obj.get_aucpr()

        return prf_test, PredScores

def tokenizeGetIDs(tokenizer, text_data, max_len):
    # Tokenize the texts using tokenizer also pad to max_len and add <cls> token to first and <sep> token to end
    input_ids = []
    attention_masks = []
    for texts in text_data:
        # text_data (n_patient, time-stamp); texts: (time-stamps, ), all times along with all notes
        Textarr = [] 
        Attnarr = []
        for text in texts:
            # text: all notes for single time stamp
            # Textarr (time-stamps, max_len)
            encoded_sent = tokenizer.encode_plus(
                    text=text,                      # Preprocess sentence
                    add_special_tokens=True,        # Add [CLS] and [SEP]
                    max_length=max_len,             # Max length to truncate/pad
                    pad_to_max_length=True,         # Pad sentence to max length
                    #return_tensors='pt',           # Return PyTorch tensor
                    return_attention_mask=True,     # Return attention mask
                    truncation=True
                    )
            Textarr.append(encoded_sent.get('input_ids'))
            Attnarr.append(encoded_sent.get('attention_mask'))

    
        # Add the outputs to the lists
        # input_ids.append(encoded_sent.get('input_ids'))
        # attention_masks.append(encoded_sent.get('attention_mask'))
        input_ids.append(Textarr)
        attention_masks.append(Attnarr)

    return input_ids, attention_masks

def concat_text_timeseries(data_reader, data_raw):

    train_text, train_times, start_time = data_reader.read_all_text_append_json(data_raw['names'], 48, NumOfNotes=NumOfNotes, notes_aggeregate = args['notes_aggeregate'])
    
    # Merge the text data with time-series data        
    data = utils.merge_text_raw(train_text, data_raw, train_times, start_time)
    return data

def get_time_to_end_diffs(times, starttimes):
    timetoends = []
    for times, st in zip(times, starttimes):
        difftimes = []
        et = np.datetime64(st) + np.timedelta64(49, 'h')

        for t in times:
            time = np.datetime64(t)
            dt = utils.diff_float(time, et)
            assert dt >= 0 #delta t should be positive
            difftimes.append(dt)
        timetoends.append(difftimes)
    return timetoends

def Read_Aggregate_data(mode, AggeragetNotesStrategies, discretizer=None, normalizer = None):
    # mode is between train, test, val
    # Build readers, discretizers, normalizers
    File_AggeragetNotesStrategies = 'Mean'
    dataPath = os.path.join('Data', mode + '_data_' +  File_AggeragetNotesStrategies + '.pkl')
    if os.path.isfile(dataPath):
        # We write the processed data to a pkl file so if we did that already we do not have to pre-process again and this increases the running speed significantly
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            (data, names, discretizer, normalizer) = pickle.load(f)
    else:
        # If we did not already processed the data we do it here
        ReaderPath = os.path.join(conf.ihm_path, 'train' if (mode == 'train') or mode == 'val' else 'test')
        reader = InHospitalMortalityReader(dataset_dir=ReaderPath,
                                                  listfile=os.path.join(conf.ihm_path, mode + '_listfile.csv'), period_length=48.0)
        
        if normalizer is None:
            discretizer = Discretizer(timestep=float(conf.timestep),
                                  store_masks=True,
                                  impute_strategy='previous',
                                  start_time='zero')
        
            discretizer_header = discretizer.transform(
                reader.read_example(0)["X"])[1].split(',')
            cont_channels = [i for (i, x) in enumerate(
                discretizer_header) if x.find("->") == -1]
            
            # text reader for reading the texts
            if (mode == 'train') or (mode == 'val'):
                text_reader = utils.TextReader(conf.textdata_fixed, conf.starttime_path)
            else:
                text_reader = utils.TextReader(conf.test_textdata_fixed, conf.test_starttime_path)
            
            # choose here which columns to standardize
            normalizer = Normalizer(fields=cont_channels)
            normalizer_state = conf.normalizer_state
            if normalizer_state is None:
                normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(
                    conf.timestep, conf.imputation)
                normalizer_state = os.path.join(
                    os.path.dirname(__file__), normalizer_state)
            normalizer.load_params(normalizer_state)
        
            normalizer = None
            # Load the patient data
        train_raw = ihm_utils.load_data(
            reader, discretizer, normalizer, conf.small_part, return_names=True)
        
        print("Number of train_raw_names: ", len(train_raw['names']))
        
        data = concat_text_timeseries(text_reader, train_raw)
        
        train_names = list(data[3])
        
        with open(dataPath, 'wb') as f:
            # Write the processed data to pickle file so it is faster to just read later
            pickle.dump((data, train_names, discretizer, normalizer), f)
        
    data_X = data[0]        # (14130, 48, 76)
    data_y = data[1]        # (14130,)
    data_text = data[2]     # (14130, n_times, note_length)
    data_names = data[3]
    data_times = data[4]    # (14130, n_times)
    start_times = data[5]   # (14130), start time in data_times
    timetoends = get_time_to_end_diffs(data_times, start_times) # (14130, n_times), from largest to smallest

    # for note length in every time stamp for all patients, avg length 228.68, std 173.16
    tokenized_path = os.path.join('Data', EmbedModelName + '_tokenized_ids_attns_' + mode + '_' + File_AggeragetNotesStrategies + '_' + str(MaxLen) + '_truncate.pkl')
    if os.path.isfile(tokenized_path):
        # If the pickle file containing text_ids exists we will just load it and save time by not computing it using the tokenizer
        with open(tokenized_path, 'rb') as f:
            txt_ids, attention_masks = pickle.load(f)
        print('txt_ids, attention_masks are loaded from Tokenize ', tokenized_path)
    else:
        txt_ids, attention_masks = tokenizeGetIDs(tokenizer, data_text, MaxLen)
        with open(tokenized_path, 'wb') as f:
            # Write the output of tokenizer to a pickle file so we can use it later
            pickle.dump((txt_ids, attention_masks), f)
        print('txt_ids, attention_masks is written to Tokenize ', tokenized_path)


    # Remove the data when text is empty
    indices = []
    for idx, txt_id in enumerate(txt_ids):
        # txt_ids: (n_patient, n_times, max_len)
        if len(txt_id) == 0:
            indices.append(idx)
        else:
            if NumOfNotes > 0:
                # Only pick the last note
                txt_ids[idx] = txt_id[-NumOfNotes:]
                attention_masks[idx] = attention_masks[idx][-NumOfNotes:]


                
    for idx in reversed(indices):
        txt_ids.pop(idx)
        attention_masks.pop(idx)
        data_X = np.delete(data_X, idx, 0)
        data_y = np.delete(data_y, idx, 0)
        data_text = np.delete(data_text, idx, 0)
        data_names = np.delete(data_names, idx, 0)
        data_times = np.delete(data_times, idx, 0)
        start_times = np.delete(start_times, idx, 0)
        timetoends = np.delete(timetoends, idx, 0)
    del data
    
    return txt_ids, attention_masks, data_X, data_y, data_text, data_names, timetoends, discretizer, normalizer

AggeragetNotesStrategies = 'Mean'

txt_ids, attention_masks, data_X, data_y, data_text, data_names, data_times, discretizer, normalizer = \
    Read_Aggregate_data('train', AggeragetNotesStrategies, discretizer=None, normalizer = None)

# txt_ids_eval, attention_masks_eval, eval_data_X, eval_data_y, eval_data_text, eval_data_names, eval_data_times, _, _ = \
#     Read_Aggregate_data('val', AggeragetNotesStrategies, discretizer=discretizer, normalizer = normalizer)

txt_ids_test, attention_masks_test, test_data_X, test_data_y, test_data_text, test_data_names, test_data_times, _, _ = \
    Read_Aggregate_data('test', AggeragetNotesStrategies, discretizer=discretizer, normalizer = normalizer)


def generate_padded_batches(x, y, t, text_ids, data_attn, data_times, batch_size):
    # Generate batches
    batches = []
    begin = 0
    while begin < len(t):            
        end = min(begin+batch_size, len(t))
        x_slice = np.stack(x[begin:end])
        y_slice = np.stack(y[begin:end])
        t_slice = np.array(text_ids[begin:end])

        attn_slice = np.array(data_attn[begin:end])
        time_slice = np.array(data_times[begin:end])
        batches.append((x_slice, y_slice, t_slice, attn_slice, time_slice))
        begin += batch_size
    return batches

def validate(model, data_X_val, data_y_val, data_text_val, txt_ids_eval, attn_val, times_val, names_eval, batch_size):
    val_batches = generate_padded_batches(data_X_val, data_y_val, data_text_val, txt_ids_eval, attn_val, times_val, batch_size)
    prf_val, probablities = Evaluate_Model(model, val_batches, names_eval)
    print(prf_val)


 
validate(FTmodel, test_data_X, test_data_y, test_data_text, txt_ids_test, attention_masks_test, test_data_times, test_data_names, batch_size)



