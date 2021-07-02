#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:06:55 2021

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
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import dvae
from dvae import LearningAlgorithm
import soundfile as sf

path_wavfile = '/data/recherche/redactions/DVAE-tutorial-icassp-2021/audio'
filename = 'SX370_f.wav'

wav_file = os.path.join(path_wavfile, filename)

options = [('VAE (complexified)', 'VAE'), 
           ('VAE (original)', 'origVAE'), 
           ('DKF', 'DKF'), 
           ('DSAE', 'DSAE'), 
           ('causal RVAE (complexified)', 'RVAE-Causal'), 
           ('non-causal RVAE (complexified)', 'RVAE-NonCausal'),
           ('causal RVAE (original)', 'origRVAE-Causal'), 
           ('non-causal RVAE (original)', 'origRVAE_NonCausal'),
           ('STORN', 'STORN'), 
           ('SRNN', 'SRNN'), 
           ('VRNN', 'VRNN')
           ]

for op in options:
    model = op[1]

    model_path = ([f[0] for f in os.walk('./saved_model') 
                    if '_'+model+'_' in f[0] and '_F' in f[0]][0])
    cfg_file = os.path.join(model_path, 'config.ini')
    print(cfg_file)
    model_state = [f for f in os.listdir(model_path) if f.endswith('.pt')]
    if len(model_state)==1:
      model_state = model_state[0]
    else:
      model_state = [tmp for tmp in model_state if 'converted' in tmp][0]
    model_state = os.path.join(model_path, model_state)
    print(model_state)
    learning_algo = LearningAlgorithm(config_file=cfg_file)
    learning_algo.load_state_dict(state_dict_file=model_state)
    
    x, sr = librosa.load(wav_file, sr=sr)
#    x = x/np.max(np.abs(x))
#    sf.write(wav_file, x, sr)   
    
    rec_wav_file = os.path.join(path_wavfile, filename[:-4] + '_' + model + '_recon.wav')
        
    learning_algo.generate(audio_orig=wav_file, audio_recon=rec_wav_file)
    
    