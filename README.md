## Exploring Slow Feature Analysis for Extracting Generative Latent Factors
This code is for the paper "Exploring Slow Feature Analysis for Generative Latent Factors" by Max Menne, Merlin Schüler, and Laurenz Wiskott to be published at [ICPRAM 2021](http://www.icpram.org).

## Overview
This repository contains the code to reproduce the experiments presented in the paper. The experiments are divided into the following files:
1. [`analyzing_reconstructability.py`](analyzing_reconstructability.py) contains the code for the experiments in Section 3.3,
2. [`latent_space_explorations.py`](latent_space_explorations.py) contains the code for the latent space explorations in Section 3.4.1,
3. [`exploring_embeddings.py`](exploring_embeddings.py) contains the code for the investigation of the embeddings in Section 3.4.1 & 3.4.2,
4. [`separated_extraction.py`](separated_extraction.py) contains the code for the separated extraction of latent factors in Section 3.4.3,
5. [`fitting_prior_distributions.py`](fitting_prior_distributions.py) contains the code for fitting the defined prior distributions in Section 3.5.1,
6. [`predicting_latent_samples.py`](predicting_latent_samples.py) contains the code for the prediction of latent samples in Section 3.5.2.

Furthermore, [`models.py`](models.py) provides the implementations of the models, [`pretrained_models`](pretrained_models) contains the trained model weights for the different experiments and [`core`](core) includes classes for the generation of several datasets as well as the implementation of the [PowerSFA framework](http://proceedings.mlr.press/v101/schuler19a.html).

## Requirements
To install requirements use
```setup
pip install -r requirements.txt
```
Further, make sure to install the newest version of the [modular-data-processing toolkit](https://github.com/mdp-toolkit/mdp-toolkit.git).

## Usage
To reproduce an experiment, simply run
```
python experiment_name.py
```
The selection and configuration of the individual models and datasets as well as the training procedure of the models can be configured within the setup section at the beginning of the respective script of each experiment.
