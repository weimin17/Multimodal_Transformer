# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers



class BioClinicalBERT_FT(nn.Module):
    # Model which only uses text part
    def __init__(self, BioBert, BioBertConfig, device):
        super(BioClinicalBERT_FT, self).__init__()
        self.BioBert = BioBert

        text_embed_size =  BioBertConfig.hidden_size
        self.FinalFC = nn.Linear(text_embed_size, 1, bias=False)
        for p in self.FinalFC.parameters():
            nn.init.xavier_uniform_(p)
            
        self.final_act = torch.sigmoid
        self.criterion= nn.BCEWithLogitsLoss()
        self.device = device
        
    def forward(self, text, attns, times):
        # For the models where we take the mean of embeddings generated for several notes we loop through the notes and take their mean
        txt_arr = []
        for txts, attn, ts in zip(text, attns, times):
            if len(txts.shape) == 1:
                # If there is only one note for a patient we just add a dimension
                txts = txts[None, :]
                attn = attn[None, :]
            txtemb = self.BioBert(txts, attn)
            emb = txtemb[0][:,0,:]
            txt_arr.append(torch.mean(emb, axis=0))

        text_embeddings = torch.stack(txt_arr)
        # deleting some tensors to free up some space
        del txt_arr


        logit_X = text_embeddings
        #Final FC layer
        logits = self.FinalFC(logit_X)
        logits = logits.squeeze(dim=-1)
        probs = self.final_act(logits)

        return logits, probs

    def get_l2(self):
        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.FinalFC.parameters():
            l2_reg += param.norm(2)
        return l2_reg

def positionalencoding1d(d_model, length):
    """
    Sinusoidal embedding
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    import math
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class multimodal_transformer(nn.Module):
    # time series on transformer, consider position embedding, use CLS
    # text, mbert
    def __init__(self, model_name, model_type, BioBert, TSModel, device):
        super(multimodal_transformer, self).__init__()

        # universal setup
        self.TSModel = TSModel
        time_series_size = 256

        self.BioBert = BioBert
        self.model_name = model_name
        self.model_type = model_type
        self.criterion= nn.BCEWithLogitsLoss()
        self.device = device


        # clinical notes
        if model_name == 'BioBert':
            print('Using BioBert model')
            text_embed_size =  BioBert.config.hidden_size 

        # clinical variables - TS
        if TSModel == 'Transformer':
            self.proj_size = 512 # project X to proj_size
            self.ts_toksample = nn.Sequential(
                nn.Linear(76, self.proj_size//2, bias=False), # transfer
                nn.ReLU(),
                nn.Linear(self.proj_size//2, self.proj_size, bias=False)
            )

            self.txt_toksample = nn.Sequential(
                nn.Linear(768, 768, bias=False)
            )
            self.emb_dim = 768
            self.toksample = nn.Sequential(
                nn.Linear(self.proj_size + 768, 1024, bias=False),
                nn.ReLU(),
                nn.Linear(1024, self.emb_dim, bias=False)
            )

            self.position_embeddings = positionalencoding1d(d_model=self.emb_dim, length=49).to(device) # (lenth, self.proj_size)
            self.LayerNorm = nn.LayerNorm(self.emb_dim) # for position embedding+ts embedding

            self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=4, dim_feedforward=time_series_size, batch_first=True) # dim_feedforward=time_series_size, 
            self.encoder_norm = nn.LayerNorm(self.emb_dim)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6, norm=self.encoder_norm)



        # Classifier
        print('Using both text and time series')
        self.FinalFC = nn.Linear(self.emb_dim + text_embed_size, 1, bias=False)
        
        
        for p in self.FinalFC.parameters():
            # Using Xavier Uniform initialization for final fully connected layer
            nn.init.xavier_uniform_(p)
        self.final_act = torch.sigmoid

        
    def forward(self, X, text, attns, times):
        # if model_type is baseline we only use the time-series part
        
        # clinical notes
        if self.model_type != 'baseline':
            if self.model_name == 'BioBert':
                ### inhour mean

                # For the models where we take the mean of embeddings generated for several notes that belong to a same hour, we loop through the notes and take their mean
                txt_arr = []
                for txts, attn, ts in zip(text, attns, times):
                    # text: (batch_size, NumOfNotes, MaxLen)
                    # attns: (batch_size, NumOfNotes, MaxLen), 0/1 indicates whether 
                    # times: (batch_size, n_times)
                    # txts: (NumOfNotes, MaxLen)
                    # attn: (NumOfNotes, MaxLen)
                    # ts: (NumOfNotes), equal to (n_times)
                    if len(txts.shape) == 1:
                        # If there is only one note for a patient we just add a dimension
                        txts = txts[None, :]
                        attn = attn[None, :]

                    txts_chunk, attn_chunk, ts_chunk = [], [], [] # (48, numofnotes, MaxLen), or (48, numofnotes)

                    ts_chunk.append( ts[ ts >= 47 ] )
                    txts_chunk.append( txts[ ts >= 47 ] )
                    attn_chunk.append( attn[ ts >= 47 ] )

                    for i in np.arange(46, -1, -1): # [46, ..., 0]
                        # ts[ torch.logical_and( i+1 > ts, ts >= i ) ]
                        ts_chunk.append( ts[ torch.logical_and( i+1 > ts, ts >= i ) ] )
                        txts_chunk.append( txts[ torch.logical_and( i+1 > ts, ts >= i ) ] )
                        attn_chunk.append( attn[ torch.logical_and( i+1 > ts, ts >= i ) ] )

                    assert(len(ts_chunk)==48)

                    txt_inhour_arr = [] # (48, 768)
                    for idx, txts_step in enumerate(txts_chunk):
                        if len(txts_step) == 0 and idx==0:
                            # TODO: what if the first step is empty
                            # use all mean as first step
                            txtemb = self.BioBert(txts[-10:], attn[-10:])
                            emb = txtemb[0][:,0,:] # (NumOfNotes, 768)
                            txt_inhour_arr.append(torch.mean(emb, axis=0))
                            continue

                        elif len(txts_step) == 0 and idx>0:
                            txt_inhour_arr.append(txt_inhour_arr[-1])
                            continue

                        attn_step = attn_chunk[idx] # (numofnotes, MaxLen)
                        ts_step = ts_chunk[idx]     # (numofnotes)
                        if len(txts_step)>=10:
                            txts_step = txts_step[-10:]
                            attn_step = attn_step[-10:]
                            ts_step = ts_step[-10:]

                        txtemb = self.BioBert(txts_step, attn_step)
                        emb = txtemb[0][:,0,:] # (NumOfNotes, 768)
                        txt_inhour_arr.append(torch.mean(emb, axis=0))
                    
                    txt_inhour_arr = torch.stack(txt_inhour_arr) # (48, 768)
                    txt_arr.append(txt_inhour_arr)
                    
                text_embeddings = torch.stack(txt_arr) # (batch_size, 48, 768)
                # deleting some tensors to free up some space
                del txt_arr


        # Getting the time-series part embedding
        ## concatenate ts with text embedding before lstm
        text_embeddings_ts = text_embeddings # (batch_size, 48, 768)
        text_embeddings_mean = torch.mean(text_embeddings, axis=1) # (batch_size, 768)
        del text_embeddings
        


        # clinical variable 
        if self.TSModel == 'Transformer':
            ## upsample
            X = self.ts_toksample(X) # (batch, 48, 76) -> (batch_size, 48, proj_size)
            text_embeddings_ts = self.txt_toksample(text_embeddings_ts) #(batch_size, 48, 768) -> (batch_size, 48, 768)

            X = torch.cat([X, text_embeddings_ts], dim = 2) # (batch_size, 48, proj_size+768)
            X = self.toksample(X) # (batch_size, 48, emb_dim)


            ## position embedding
            position_embeddings = self.position_embeddings.unsqueeze(0) # (1, 49, emb_dimn)
            position_embeddings = position_embeddings.repeat(X.shape[0], 1, 1) # (batch_size, 49, emb_dimn)
            ## add cls tok to X
            cls = torch.zeros(size=(X.size(0), 1, X.size(2) ), dtype = X.dtype, device=X.device) #(batch_size, 1, emb_dimn)
            X = torch.cat([cls, X], dim=1) # (batch_size, 49, emb_dimn)
            ## add position embedding
            X += position_embeddings
            embeddings = self.LayerNorm(X)

            rnn_outputs = self.transformer_encoder(embeddings) # (batch_size, 49, self.emb_dim)
            mean_rnn_outputs = rnn_outputs[:, 0, :] # (batch_size, self.emb_dim)


        # Classifier - Final FC layer
        # Concatenating time-series and text embedding
        logit_X = torch.cat((text_embeddings_mean.float(), mean_rnn_outputs.float()), 1)
        logits = self.FinalFC( logit_X )
        logits = logits.squeeze(dim=-1)
        probs = self.final_act(logits)
        return logits, probs
    
    def get_l2(self):
        # get l2 regularization weight of the cnn and final layer
        l2_reg = torch.tensor(0.).to(self.device)

        for param in self.FinalFC.parameters():
            l2_reg += param.norm(2)
        return l2_reg

