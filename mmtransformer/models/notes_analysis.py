'''
Generate important clinical notes words. 

'''
import enum
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

from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel

import warnings
import time
from captum.attr import visualization


# Ignoring warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', help='Which gpu to use', default='7', type=str)

args = vars(parser.parse_args())
conf = utils.get_config() # Get configs

start_time = time.time()

os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_id']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Loading pre-trained model based on EmbedModel argument
EmbedModelName = "BioBert"
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
BioBert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

BioBertConfig = BioBert.config


TextModelCheckpoint = 'BioClinicalBERT_FT'
batch_size = 100
NumOfNotes=10
output_file_name = 'analysis'
MaxLen=128


# Use text model checkpoint to update the bert model to fine-tuned weights
model = torch.load(os.path.join('Checkpoints', 'BioClinicalBERT_FT')) 
model = model.to(device)
model.eval()
model.zero_grad()



import torch
import torch.nn as nn

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig

from captum.attr import IntegratedGradients
from captum.attr import TokenReferenceBase
from captum.attr import visualization


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# We need to split forward pass into two part: 
# 1) embeddings computation
# 2) classification

def compute_bert_outputs(model_bert, embedding_output, attention_mask=None, head_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    extended_attention_mask = extended_attention_mask.to(dtype=next(model_bert.parameters()).dtype) # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(model_bert.config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        head_mask = head_mask.to(dtype=next(model_bert.parameters()).dtype) # switch to fload if need + fp16 compatibility
    else:
        head_mask = [None] * model_bert.config.num_hidden_layers

    encoder_outputs = model_bert.encoder(embedding_output,
                                         extended_attention_mask,
                                         head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = model_bert.pooler(sequence_output)
    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)    


class BertModelWrapper(nn.Module):
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, embeddings):        
        outputs = compute_bert_outputs(self.model.BioBert, embeddings)
        pooled_output = outputs[1]
        # pooled_output = self.model.dropout(pooled_output)
        logits = self.model.FinalFC(pooled_output)
        final_output = torch.sigmoid(logits.squeeze(dim=-1))
        return final_output #torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)

    
bert_model_wrapper = BertModelWrapper(model)
ig = IntegratedGradients(bert_model_wrapper)


def interpret_sentence(model_wrapper, input_ids, vis_data_records_ig, tokenlist_top10, tokenlist_bot10, label=1):

    model_wrapper.eval()
    model_wrapper.zero_grad()
    
    # input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])
    # input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True,max_length=16,pad_to_max_length=True,truncation=True)])
    # #return_tensors='pt',           # Return PyTorch tensor
    # return_attention_mask=True,     # Return attention mask

    input_embedding = model_wrapper.model.BioBert.embeddings(input_ids)
    
    # predict
    pred = model_wrapper(input_embedding).item()
    pred_ind = round(pred)

    # compute attributions and approximation delta using integrated gradients
    attributions_ig, delta = ig.attribute(input_embedding, n_steps=50, return_convergence_delta=True)

    with open('Analysis/bert_analysis_pred2.txt', 'a+') as f:
        # print('pred: ', pred_ind, '(', '%.2f' % pred, ')', ', delta: ', abs(delta), file=f)
        print('pred: {}( {:.2f} ), delta: {}'.format(pred_ind, pred, abs(delta).cpu().numpy()[0]), file=f )
        f.close()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().numpy().tolist())    
    add_attributions_to_visualizer(attributions_ig, tokens, pred, pred_ind, label, delta, vis_data_records_ig, tokenlist_top10, tokenlist_bot10)
    
    
def add_attributions_to_visualizer(attributions, tokens, pred, pred_ind, label, delta, vis_data_records, tokenlist_top10, tokenlist_bot10):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().cpu().numpy()
    # print('shape attributions', np.shape(attributions), max(attributions), min(attributions))
    ind_top10 = np.argpartition(attributions, -10)[-10:]
    ind_bot10 = np.argpartition(-attributions, -10)[-10:]
    tokens_top10 = np.array(tokens)[ind_top10].tolist()
    tokens_bot10 = np.array(tokens)[ind_bot10].tolist()
    tokenlist_top10.append(tokens_top10)
    tokenlist_bot10.append(tokens_bot10)
    
    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            pred_ind,
                            label,
                            "label",
                            attributions.sum(),       
                            tokens[:len(attributions)],
                            delta))    




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
        print('File Missing')
        exit(0)
        
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
        print('File missing. ')
        exit(0)


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



def Attribution_Model(model, batch):
    # Test the model
    model.eval()
    # accumalate couple samples in this array for visualization purposes
    vis_data_records_ig_l0, tokenlist_top10_l0, tokenlist_bot10_l0 = [], [], []
    vis_data_records_ig_l1, tokenlist_top10_l1, tokenlist_bot10_l1 = [], [], []

    with torch.no_grad():

        for idx, sample in enumerate(batch):
            # X = torch.tensor(sample[0], dtype=torch.float).to(device)
            y = sample[1] # (batch_size)
            text = [torch.tensor(x, dtype=torch.long).to(device) for x in sample[2]] # (batch_size, 10, max_len) token_id, not txt
            # attn = [torch.tensor(x, dtype=torch.long).to(device) for x in sample[3]]
            # times = [torch.tensor(x, dtype=torch.long).to(device) for x in sample[4]]

            if sample[2].shape[0] == 0:
                continue
            # Run the model
            for count, single_y in enumerate(y):
                single_visit = text[count] # (10, max_length)
                for count2, single_text in enumerate(single_visit):
                    single_input_id = single_text.unsqueeze(0) # (1, max_len)
                    # print('single_y', single_y, single_y==0, single_y==1)
                    input_embedding = bert_model_wrapper.model.BioBert.embeddings(single_input_id)
                    pred = bert_model_wrapper(input_embedding).item()
                    pred_ind = round(pred)

                    ## list positive and negative patients together. Only list the correct predicted patient
                    if pred_ind == single_y:
                        interpret_sentence(bert_model_wrapper, input_ids=single_input_id, vis_data_records_ig=vis_data_records_ig_l0, tokenlist_top10=tokenlist_top10_l0, tokenlist_bot10=tokenlist_bot10_l0, label=single_y)

            if idx % 1 == 0: # save checkpoint
                with open('Analysis/bert_analysis_pred2_step_{}.pkl'.format(idx), 'wb') as handle:
                    pickle.dump([vis_data_records_ig_l0, tokenlist_top10_l0, tokenlist_bot10_l0, vis_data_records_ig_l1, tokenlist_top10_l1, tokenlist_bot10_l1], handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Analysis/bert_analysis_pred_all2.pkl', 'wb') as handle:
        pickle.dump([vis_data_records_ig_l0, tokenlist_top10_l0, tokenlist_bot10_l0, vis_data_records_ig_l1, tokenlist_top10_l1, tokenlist_bot10_l1], handle, protocol=pickle.HIGHEST_PROTOCOL)




def validate(model, data_X_val, data_y_val, data_text_val, txt_ids_eval, attn_val, times_val, batch_size):
    val_batches = generate_padded_batches(data_X_val, data_y_val, data_text_val, txt_ids_eval, attn_val, times_val, batch_size)
    Attribution_Model(model, val_batches)


validate(model, test_data_X, test_data_y, test_data_text, txt_ids_test, attention_masks_test, test_data_times, batch_size)



