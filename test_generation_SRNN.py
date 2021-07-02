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

#%%
model_dir = './saved_model/WSJ0_2020-08-12-21h00_SRNN_z_dim=16_F'

# find config file and training weight
for file in os.listdir(model_dir):
    if '.ini' in file:
        cfg_file = os.path.join(model_dir, file)
    if 'final_epoch' in file:
        weight_file = os.path.join(model_dir, file)

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

#%%

def sample_srnn(seq_len=300, x_dim=257, z_dim=16, h_dim=128, sample_x=True):

    h_t = torch.zeros(1,1,h_dim).to(device)
    c_t = torch.zeros(1,1,h_dim).to(device)
    # you can try different random init for z_t
    z_t = torch.randn(1,1,z_dim).to(device)
    x_t = torch.abs(torch.randn(1,1,x_dim).to(device))
    
    x_all = np.zeros((seq_len, x_dim))
    
    for t in np.arange(0,seq_len):
        
        # deterministic h
        x_h = model.mlp_x_h(x_t)
        _, (h_t, c_t) = model.rnn_h(x_h, (h_t, c_t))
        
        # generation z
        hz_z = torch.cat((h_t, z_t), -1)
        hz_z = model.mlp_hz_z(hz_z)
        z_mean_p_t = model.prior_mean(hz_z)
        z_logvar_p_t = model.prior_logvar(hz_z)
        z_t = model.reparameterization(z_mean_p_t, z_logvar_p_t)
        
        # generation x
        hz_x = torch.cat((h_t, z_t), -1)
        hz_x = model.mlp_hz_x(hz_x)
        logvar_x_t = model.gen_logvar(hz_x)
        var_x_t = torch.exp(logvar_x_t)
        
        if sample_x:
            # sample the complex gaussian distribution
            x_t_cplx_r = torch.sqrt(var_x_t/2)*torch.randn_like(var_x_t).to(device) 
            x_t_cplx_i = torch.sqrt(var_x_t/2)*torch.randn_like(var_x_t).to(device) 
            x_t = x_t_cplx_r**2 + x_t_cplx_i**2
        else:
            # or simply reinject the variance
            x_t = var_x_t
        
        
        x_all[t,:] = x_t.detach().cpu().numpy()
        
    return x_all.T

#%% pure generation

seq_len = 288

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

for n in np.arange(20):

    power_spec = sample_srnn(seq_len=seq_len, x_dim=x_dim, z_dim=z_dim, 
                             h_dim=h_dim, sample_x=False)
    
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
    
    figure_file = '/data/tmp/gen_speech_SRNN/gen_speech_srnn_'+ str(n+1) + '.png'
    plt.savefig(figure_file) 
    
    sf.write('/data/tmp/gen_speech_SRNN/gen_speech_srnn_'+ str(n+1) + '.wav', s_inv, fs)
    
    plt.close()