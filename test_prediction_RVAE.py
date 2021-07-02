#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:14:22 2020

@author: sleglaive
"""

import os
import numpy as np
import torch
from dvae.utils import myconf
import librosa
import librosa.display
import soundfile as sf
from dvae.model import build_VAE, build_DKF, build_KVAE, build_STORN, build_VRNN, build_SRNN, build_RVAE, build_DSAE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.close('all')

#%%

#model_dir = './saved_model/WSJ0_2019-07-15-10h01_origRVAE_NonCausal_latent_dim=16_F'
model_dir = './saved_model/WSJ0_2020-09-29-14h48_RVAE-NonCausal_z_dim=16'

# find config file and training weight
cfg_file = os.path.join(model_dir, 'config.ini')
model_state = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
if len(model_state)==1:
  model_state = model_state[0]
else:
  model_state = [tmp for tmp in model_state if 'converted' in tmp][0]
weight_file = os.path.join(model_dir, model_state)

# read config file
cfg = myconf()
cfg.read(cfg_file)
model_name = cfg.get('Network', 'name')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# build model
if model_name == 'VAE':
    model = build_VAE(cfg=cfg, device=device)
elif model_name == 'DMM':
    model = build_DKF(cfg=cfg, device=device)
elif model_name == 'STORN':
    model = build_STORN(cfg=cfg, device=device)
elif model_name == 'VRNN':
    model = build_VRNN(cfg=cfg, device=device)
elif model_name == 'SRNN':
    model = build_SRNN(cfg=cfg, device=device)
elif model_name == 'RVAE':
    model = build_RVAE(cfg=cfg, device=device)
elif model_name == 'DSAE':
    model = build_DSAE(cfg=cfg, device=device)
elif model_name == 'KVAE':
    model = build_KVAE(cfg=cfg, device=device)
    
# Load weight
model.load_state_dict(torch.load(weight_file, map_location=device))
model.eval()


#%% pure generation

x_dim = cfg.getint('Network', 'x_dim')
z_dim = cfg.getint('Network', 'z_dim')
h_dim = cfg.getint('Network', 'dim_RNN_h')

fs = cfg.getint('STFT', 'fs')
wlen_sec = cfg.getfloat('STFT', 'wlen_sec')
hop_percent = cfg.getfloat('STFT', 'hop_percent')
zp_percent = cfg.getfloat('STFT', 'zp_percent')
wlen = wlen_sec*fs # window length in samples
wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
hop = np.int(hop_percent*wlen) # hop size in samples
nfft = wlen + int(zp_percent*wlen) # number of points of the discrete Fourier transform
win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi) # sine analysis window


#%% analysis-resynthesis then prediction

file_list = librosa.util.find_files('/data/datasets/clean_speech/wsj0_si_dt_05', ext='wav')


n_files = 50

for n in np.arange(n_files):
    
    ind = np.random.randint(low=0, high=len(file_list))
    wavfile = file_list[ind]
    
    x, fs_x = sf.read(wavfile) 
    x = x/np.max(np.abs(x))
    x, _ = librosa.effects.trim(x, top_db=30)
    
    x_orig = x
    x = x[:int(2*fs)]
    
    X = librosa.stft(x, n_fft=nfft, hop_length=hop, 
                                 win_length=wlen,
                                 window=win) # STFT
    
    F, N = X.shape
    
    # Prepare data input
    data_orig = np.abs(X) ** 2 # (x_dim, seq_len)
    data_orig = torch.from_numpy(data_orig.astype(np.float32)).to(device)
        
    #%% forward
    
    sample_x = False
    
    with torch.no_grad():
    
        x = data_orig
            
        # train input: (batch_size, x_dim, seq_len)
        # test input:  (x_dim, seq_len)
        # need input:  (seq_len, batch_size, x_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.permute(-1, 0, 1)

        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # main part
        z, z_mean, z_logvar = model.inference(x)
        y = model.generation_x(z)
        data_recon = y.cpu().numpy()
    
    
    #%% Precition
            
        z = torch.cat((z, torch.randn(seq_len, 1, z_dim)), dim=0)
        
        # 1. z_t to h_t
        z_h = model.mlp_z_h(z)
        
        # 2. h_t recurrence
        h, _ = model.rnn_h(z_h)
        
        # 3. h_t to y_t
        hx = model.mlp_h_x(h)
        logvar_x = model.gen_logvar(hx)
        var_x = torch.exp(logvar_x).squeeze()
        
        if sample_x:
            # sample the complex gaussian distribution
            x_cplx_r = torch.sqrt(var_x/2)*torch.randn_like(var_x).to(device) 
            x_cplx_i = torch.sqrt(var_x/2)*torch.randn_like(var_x).to(device) 
            x = x_cplx_r**2 + x_cplx_i**2
        else:
            # or simply reinject the variance
            x = var_x
            
        x = x.detach().cpu().numpy()
        
#%%
            
    power_spec = x.T    
    
    mag_spec = np.sqrt(power_spec)
        
        
        
    s_inv = librosa.griffinlim(mag_spec, n_iter=100, hop_length=hop, 
                               win_length=wlen, window=win)
    
    plt.figure(figsize=(20,15))
    
    plt.subplot(2,1,1)
    librosa.display.specshow(librosa.power_to_db(power_spec), sr=fs, 
                                                 hop_length=hop, 
                                                 y_axis='linear', 
                                                 x_axis='time')
    plt.subplot(2,1,2)
    time_axis = np.arange(0,s_inv.shape[0])/fs
    plt.plot(time_axis, s_inv)
    plt.xlim([time_axis[0], time_axis[-1]])
    
    
    plt.plot(np.array([seq_len, seq_len])*hop/fs, np.array([-1, 1]))
    
    figure_file = '/data/tmp/pred_speech_RVAE/pred_speech_rvae_'+ str(n+1) + '.png'
    plt.savefig(figure_file) 
    
    sf.write('/data/tmp/pred_speech_RVAE/pred_speech_rvae_'+ str(n+1) + '.wav', s_inv, fs)
    
    plt.close()