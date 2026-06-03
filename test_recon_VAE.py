#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:14:22 2020

@author: sleglaive
"""

import os
import numpy as np
np.random.seed(0)
import torch
from dvae.utils import myconf
import librosa
import librosa.display
import soundfile as sf
from dvae.model import build_VAE, build_DKF, build_KVAE, build_STORN, build_VRNN, build_SRNN, build_RVAE, build_DSAE
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.close('all')

#%%

model_dir = './saved_model/WSJ0_2019-07-15-10h21_origVAE_latent_dim=16_F'


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
model = model.to(device)

#%% pure generation

x_dim = cfg.getint('Network', 'x_dim')
z_dim = cfg.getint('Network', 'z_dim')

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


n_files = 5

for n in np.arange(n_files):
    
    ind = np.random.randint(low=0, high=len(file_list))
    wavfile = file_list[ind]
    
    x, fs_x = sf.read(wavfile) 
    x = x/np.max(np.abs(x))
    x, _ = librosa.effects.trim(x, top_db=30)
    
    x_orig = x
    # x = x[:int(2*fs)]
    
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
        y = model.generation_x(z_mean)
    

    
            
#%%
    
    data_recon = y.cpu().numpy().squeeze()
    data_recon = data_recon.T
    data_orig = data_orig.cpu().numpy()
    
    
    X_recon = np.sqrt(data_recon)*np.exp(1j*np.angle(X))
    x_recon = librosa.istft(X_recon, hop_length=hop, win_length=wlen, window=win)
    
    scale = 1/(np.maximum(np.max(np.abs(x_recon)),np.max(np.abs(x_orig))))*0.9
    
    
    sf.write('/data/tmp/rec_speech_vae_'+ str(n+1) + '.wav', scale*x_recon, fs)
    sf.write('/data/tmp/orig_speech_vae_'+ str(n+1) + '.wav', scale*x_orig, fs)
    

    c_max = np.max((np.max(10*np.log10(data_orig)), np.max(10*np.log10(data_recon))))
    c_min = c_max - 80
    
    
    fig = plt.figure(figsize=(10, 15))
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[3, 1, 3])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # --- Top plot ---
    img1 = librosa.display.specshow(
        librosa.power_to_db(data_orig),
        sr=fs,
        hop_length=hop,
        y_axis='linear',
        x_axis='time',
        cmap='magma',
        ax=ax1
    )
    # ax1.set_cmap('magma')
    img1.set_clim(c_min, c_max)
    ax1.set_ylabel('frequency (Hz)', fontsize=24)
    ax1.set_xticks([])
    ax1.set_xlabel('', fontsize=24)
    ax1.tick_params(labelsize=16)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="1.5%", pad=0.1) 
    cb1 = fig.colorbar(img1, cax=cax1)
    cb1.ax.tick_params(labelsize=16)
    ax1.set_title('Original spectrogram', fontsize=24)
    
    # --- Middle (smaller) plot ---
    img2 = ax2.imshow(
        z_mean.detach().cpu().numpy().squeeze().T,
        origin='lower',
        aspect='auto',
        cmap='magma'
    )
    
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="1.5%", pad=0.1) 
    fig.colorbar(img2, cax=cax2)
    cb2 = fig.colorbar(img2, cax=cax2)
    cb2.ax.tick_params(labelsize=16)
    ax2.set_xticks([])
    ax2.tick_params(labelsize=16)
    ax2.set_ylabel('latent dim.', fontsize=24)
    ax2.set_title('Latent representation', fontsize=24)
    
    # --- Bottom plot ---
    img3 = librosa.display.specshow(
        librosa.power_to_db(data_recon),
        sr=fs,
        hop_length=hop,
        y_axis='linear',
        x_axis='time',
        cmap='magma',
        ax=ax3
    )
    # ax3.set_cmap('magma')
    img3.set_clim(c_min, c_max)
    ax3.set_ylabel('frequency (Hz)', fontsize=24)
    ax3.set_xlabel('time (s)', fontsize=24)
    ax3.tick_params(labelsize=16)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="1.5%", pad=0.1) 
    fig.colorbar(img3, cax=cax3)
    cb3 = fig.colorbar(img3, cax=cax3)
    cb3.ax.tick_params(labelsize=16)
    ax3.set_title('Reconstructed spectrogram', fontsize=24)
    
    # fig.suptitle(model_name, fontsize=24)
    fig.tight_layout()
    
    
    figure_file = '/data/tmp/rec_speech_vae_'+ str(n+1) + '.pdf'
    fig.savefig(figure_file)
    plt.close(fig)
    
    