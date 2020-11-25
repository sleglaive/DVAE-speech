# Dynamical Variational Autoencoders

This repository contains the code for this paper:

**Dynamical Variational Autoencoders: A Comprehensive Review**  
[Laurent Girin](http://www.gipsa-lab.grenoble-inp.fr/~laurent.girin/cv_en.html), [Simon Leglaive](https://sleglaive.github.io/index.html), [Xiaoyu Bie](https://team.inria.fr/perception/team-members/xiaoyu-bie/), [Julien Diard](https://diard.wordpress.com/), [Thomas Hueber](http://www.gipsa-lab.grenoble-inp.fr/~thomas.hueber/), [Xavier Alameda-Pineda](http://xavirema.eu/)  
**[[Paper](https://hal.inria.fr/hal-02926215)]**

More precisely, it is a re-implementation of the following models in Pytorch for speech re-synthesis task:
- [VAE](https://arxiv.org/abs/1312.6114)
- [DKF](https://arxiv.org/abs/1609.09869)
- [KVAE](https://papers.nips.cc/paper/6951-a-disentangled-recognition-and-nonlinear-dynamics-model-for-unsupervised-learning)
- [STORN](https://arxiv.org/abs/1411.7610)
- [VRNN](https://arxiv.org/abs/1506.02216)
- [SRNN](https://arxiv.org/abs/1605.07571)
- [RVAE](https://arxiv.org/abs/1910.10942)
- [DSAE](https://arxiv.org/abs/1803.02991)


## Bibtex
If you find this code useful, please star the project and consider citing:

```
@article{dvae2020,
  title={Dynamical Variational Autoencoders: A Comprehensive Review},
  author={Girin, Laurent and Leglaive, Simon and Bie, Xiaoyu and Diard, Julien and Hueber, Thomas and Alameda-Pineda, Xavier},
  Journal={arXiv preprint arXiv:2008.12595},
  year={2020}
}

```


## Installation instructions


This code has been tested on Ubuntu 16.04, Python 3.7, Pytorch=1.3.1, CUDA 9.2, NVIDIA Titan X and NVIDIA Titan RTX, you could use it in the following methods:


### Install as a package
The simplest way to use our code is install it as a python package:

```bash
pip install git+https://gitlab.inria.fr/xbie/dvae-speech
```

, then you could import it like other packages:

```python
import dvae
```

, for more detailed usage tutorial, pleas find in section **Example**


### Use container
It's highly recommended to use Singularity container for results reproducing or further application, we provide a sigularity definition file [singularity/dvae.def](https://gitlab.inria.fr/xbie/dvae-speech/-/blob/master/singularity/dvae.def) to build the images:

```bash
# Download singularity
sudo apt-get install -y singularity-container

# Build singularity image
sudo singularity build dvae.sif example_singularity/dvae.def

# Shell into the image, no cuda
singularity shell --bind /path_to_your_dir/:/mnt/your_dir_name singularity/dvae.sif

# Execute commands, enable cuda, you need to define the data path in the config files
singularity exec --nv --bind /path_to_your_dir/:/mnt singularity/dvae.sif python train_model.py example_configuration/cfg_dkf.ini
```

For more information about singularity, please read [Singularity User Guide](https://singularity-userdoc.readthedocs.io/en/latest/)

### Tips

- For those who want to use this package on `MacOS`, please go to the branch `code_for_mac`, that branch omits all function about speech evaluation, since there exist some compilation errors for `pypesq` on `MacOS`
- Python 3.8 support requires Tensorflow 2.2 or later ([link](https://www.tensorflow.org/install/pip)), so it is recommended to use Python 3.7 since `speechmetrics` package needs `Tensorflow 2.0`


## Usage

`dvae` has been designed to be designed to be easily used in a modular way. All you need to do is to specify a configure file path to innitiale a model, then you could either:

- train(): train your model with training/validation dataset specified in your configure file
- generate(): generate a reconstructed audio with an input audio file
- eval(): evaluate reconstrcuted audio file with specified metric (`rmse`, `pesq`, `stoi`, `all`) 
- test(): apply reconstruction for all audio files with indicated test dataset directory path

### Example

```python
# instantiate a model
from dvae import LearningAlgorithm
cfg_file = 'config.ini'
learning_algo = LearningAlgorithm(config_file=cfg_file)

# train your model
learning_algo.train()

# generate audio with model state in cache
# reconstructed audio will be saved in the same path as input audio file, named as 'audio_001_recon.wav'
audio_ref = 'audio_001.wav'
learning_algo.generate(audio_orig=audio_ref, audio_recon=None, state_dict_file=None)

# generate audio with given model state
# reconstructed audio will be saved in the given path
audio_recon = 'recon/audio_001_recon.wav'
model_state = 'model_state.pt'
learning_algo.generate(audio_orig=audio_ref, audio_recon=audio_recon, state_dict_file=model_state)

# evaluate audio quality with model state in cache
score_rmse = learning_algo.eval(audio_ref=audio_ref, audio_est=audio_recon, metric='rmse') # only RMSE
score_rmse, score_pesq, score_stoi = learning_algo.eval(audio_ref=audio_ref, audio_est=audio_recon, metric='all') # both RMSE, PESQ and STOI

# test model on test dataset with model state in cache
test_data_dir = 'data_to_test'
list_score_rmse, list_score_pesq, list_score_stoi = learning_algo.test(data_dir=test_data_dir, state_dict_file=None)
```

### Config file

We provide all configuration examples of the above models in [config/](https://gitlab.inria.fr/xbie/dvae-speech/-/tree/master/config). For results reproducing, all you need to do is to replace `saved_root`, `train_data_dir` and `val_data_dir` with your own directory path 


## Main results

All models are trained with Adam with a batch size of 32, using:

- training dataset: wsj0_si_tr_s
- validation dataset: wsj0_si_dt_05
- test dataset: wsj0_si_et_05

For more details, pleas visit **Chapter 13 Experiments** in our article.

We propose the evaluation results in average and their training curve:

| DVAE           |  RMSE  | PESQ | STOI |
| ----           |  ----  | ---- | ---- |
| VAE            | 0.0510 | 2.05 | 0.86 |
| DKF            | 0.0344 | 3.30 | 0.94 |
| STORN          | 0.0338 | 3.05 | 0.93 |
| VRNN           | 0.0267 | 3.60 | 0.96 |
| SRNN           | 0.0248 | 3.64 | 0.97 |
| RVAE-Causal    | 0.0499 | 2.27 | 0.89 |
| RVAE-NonCausal | 0.0479 | 2.37 | 0.89 |
| DSAE           | 0.0469 | 2.32 | 0.90 |

#### VAE
![vae](https://gitlab.inria.fr/xbie/dvae-speech/-/raw/master/figures/loss_VAE.png)

#### DKF
![dkf](https://gitlab.inria.fr/xbie/dvae-speech/-/raw/master/figures/loss_DKF.png)

#### STORN
![storn](https://gitlab.inria.fr/xbie/dvae-speech/-/raw/master/figures/loss_STORN.png)

#### VRNN
![vrnn](https://gitlab.inria.fr/xbie/dvae-speech/-/raw/master/figures/loss_VRNN.png)

#### SRNN
![srnn](https://gitlab.inria.fr/xbie/dvae-speech/-/raw/master/figures/loss_SRNN.png)

#### RVAE-Causal
![rvae-causal](https://gitlab.inria.fr/xbie/dvae-speech/-/raw/master/figures/loss_RVAE-Causal.png)

#### RVAE-NonCausal
![rvae-noncausal](https://gitlab.inria.fr/xbie/dvae-speech/-/raw/master/figures/loss_RVAE-NonCausal.png)

#### DSAE
![dsae](https://gitlab.inria.fr/xbie/dvae-speech/-/raw/master/figures/loss_DSAE.png)

