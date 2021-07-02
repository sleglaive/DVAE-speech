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
        x_dim = x.shape[2]
        
        # main part
        x_0 = torch.zeros(1, batch_size, x_dim).to(model.device)
        x_tm1 = torch.cat((x_0, x[:-1, :, :]), 0)
        
        x_h = model.mlp_x_h(x_tm1)
        h, (h_t, c_t) = model.rnn_h(x_h)
        
        z, z_mean, z_logvar = model.inference(x, h)
        z_0 = torch.zeros(1, batch_size, model.z_dim).to(model.device)
        z_tm1 = torch.cat((z_0, z[:-1, :, :]), 0)
        z_mean_p, z_logvar_p = model.generation_z(h, z_tm1)
        y = model.generation_x(z, h)
        
        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)
        model.y = y.permute(1,-1,0).squeeze()
        model.z = z.permute(1,-1,0).squeeze()
        model.z_mean = z_mean.permute(1,-1,0).squeeze()
        model.z_logvar = z_logvar.permute(1,-1,0).squeeze()
        model.z_mean_p = z_mean_p.permute(1,-1,0).squeeze()
        model.z_logvar_p = z_logvar_p.permute(1,-1,0).squeeze()
        
        data_recon = model.y.cpu().numpy()
    
    
    #%% Precition
    
    sample_x = False
    reset_RNN_internal_state = False
    
    # you can try different random init for z_t
    z_t = z[-1,0,:].view(1,1,z_dim)
    x_t = x[-1,0,:].view(1,1,x_dim)
    
    if reset_RNN_internal_state:
        h_t = torch.zeros(1,1,h_dim).to(device)
        c_t = torch.zeros(1,1,h_dim).to(device)
    
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
    
    #%%
    
    power_spec = x_all.T
    
    power_spec = np.hstack((data_recon, power_spec))
    
    
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
    
    figure_file = '/data/tmp/pred_speech_SRNN/pred_speech_srnn_'+ str(n+1) + '.png'
    plt.savefig(figure_file) 
    
    sf.write('/data/tmp/pred_speech_SRNN/pred_speech_srnn_'+ str(n+1) + '.wav', s_inv, fs)
    
    plt.close()