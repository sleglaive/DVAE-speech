#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:44:08 2021

@author: simon
"""

my_seed = 0
import numpy as np
np.random.seed(my_seed)
import torch
torch.manual_seed(my_seed)
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display

sr = 16000

wlen_sec = 64e-3 # STFT window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
wlen = wlen_sec*sr # window length of 64 ms
wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
hop = np.int(hop_percent*wlen) # hop size
win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window

device ='cuda:0'

#%% Load VAE Xiaoyu

from dvae import LearningAlgorithm


cfg_file = '/data/recherche/python/DVAE-speech/saved_model/WSJ0_2019-07-15-10h21_origVAE_latent_dim=16_F/config.ini'

learning_algo = LearningAlgorithm(config_file=cfg_file)

for param_tensor in learning_algo.model.state_dict():
    print(param_tensor, "\t", learning_algo.model.state_dict()[param_tensor].size())

#%% Load VAE Simon

from VAEs_simon import VAE

input_dim = 513
latent_dim = 16   
hidden_dim_encoder = [128]
activation = torch.tanh

vae = VAE(input_dim=input_dim, latent_dim=latent_dim, 
            hidden_dim_encoder=hidden_dim_encoder,
            activation=activation).to('cuda:0')

saved_model = '/data/recherche/python/DVAE-speech/saved_model/WSJ0_2019-07-15-10h21_origVAE_latent_dim=16_F/final_model_RVAE_epoch65.pt'
vae.load_state_dict(torch.load(saved_model, map_location=device))


for param_tensor in vae.state_dict():
    print(param_tensor, "\t", vae.state_dict()[param_tensor].size())
    
#%% Transfer Simon's VAE parameters to Xiaoyu's and save
    
layers_xiaoyu = ['mlp_x_z.linear0.weight',
'mlp_x_z.linear0.bias',
'inf_mean.weight',
'inf_mean.bias',
'inf_logvar.weight',
'inf_logvar.bias',
'mlp_z_x.linear0.weight',
'mlp_z_x.linear0.bias',
'gen_logvar.weight',
'gen_logvar.bias'
]

layers_simon = ['encoder_layers.0.weight',
'encoder_layers.0.bias',
'latent_mean_layer.weight',
'latent_mean_layer.bias',
'latent_logvar_layer.weight',
'latent_logvar_layer.bias',
'decoder_layers.0.weight',
'decoder_layers.0.bias',
'output_layer.weight',
'output_layer.bias'
]


#%% change weights in Xiaoyu's VAE state dict

simon_vae_state_dict = vae.state_dict()
xiaoyu_vae_state_dict = learning_algo.model.state_dict()

for ind, key in enumerate(layers_xiaoyu):
    xiaoyu_vae_state_dict[key] = simon_vae_state_dict[layers_simon[ind]]
    
save_file = '/data/recherche/python/DVAE-speech/saved_model/WSJ0_2019-07-15-10h21_origVAE_latent_dim=16_F/final_model_RVAE_epoch65_converted.pt'
torch.save(xiaoyu_vae_state_dict, save_file)

#%% test

learning_algo = LearningAlgorithm(config_file=cfg_file)
learning_algo.load_state_dict(state_dict_file=save_file)

for ind, key in enumerate(layers_xiaoyu):
    print(torch.mean(learning_algo.model.state_dict()[key]) == torch.mean(simon_vae_state_dict[layers_simon[ind]]))

#%%


audio_orig = '/data/datasets/clean_speech/wsj0_si_dt_05/050/050a050c.wav'
audio_recon = './050a050c_rec_x.wav'


import soundfile as sf
# Read original audio file
x, fs_x = sf.read(audio_orig)

# Scaling
scale = np.max(np.abs(x))
x = x / scale

# STFT
X = librosa.stft(x, n_fft=wlen, hop_length=hop, win_length=wlen, window=win)

# Prepare data input        
data_orig = np.abs(X) ** 2 # (x_dim, seq_len)
data_orig = torch.from_numpy(data_orig.astype(np.float32)).to(learning_algo.device) 
        
# Set module.training = False
learning_algo.model.eval()

# Reconstruction
with torch.no_grad():
    data_recon = learning_algo.model(data_orig, compute_loss=False).to('cpu').detach().numpy()

# Re-synthesis
X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(X))
x_recon = librosa.istft(X_recon, hop_length=hop, win_length=wlen, window=win, length=x.shape[0])

# Wrtie audio file
sf.write(audio_recon, scale*x_recon, fs_x)


#%%

audio_orig = '/data/datasets/clean_speech/wsj0_si_dt_05/050/050a050c.wav'
audio_recon = './050a050c_rec_s.wav'


import soundfile as sf
# Read original audio file
x, fs_x = sf.read(audio_orig)

# Scaling
scale = np.max(np.abs(x))
x = x / scale

# STFT
X = librosa.stft(x, n_fft=wlen, hop_length=hop, win_length=wlen, window=win)

# Prepare data input        
data_orig = np.abs(X) ** 2 # (x_dim, seq_len)
        
# Set module.training = False
vae.eval()

# Reconstruction
with torch.no_grad():
    
    data_orig = data_orig.T
    data_orig = torch.from_numpy(data_orig.astype(np.float32))
    data_orig = data_orig.to(device)
    
    data_recon, mean, logvar, z = vae(data_orig)
    mean = mean.detach().cpu().numpy().T
    data_recon = data_recon.detach().cpu().numpy().T
    
    data_orig = data_orig.detach().cpu().numpy().T

# Re-synthesis
X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(X))
x_recon = librosa.istft(X_recon, hop_length=hop, win_length=wlen, window=win, length=x.shape[0])

# Wrtie audio file
sf.write(audio_recon, scale*x_recon, fs_x)






















