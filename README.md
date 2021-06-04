# MathONet

This repository is the implementation of [Bayesian Learning to Discover Mathematical Operations in Governing Equations of Dynamic Systems]

## Description

This paper works on...

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Several hyperparameters needs to be defined in the train.py, including: 
1. Number of hidden layers and hidden neurons for MathONet; 
2. Initialization of unary operations.
3. Value of regularization paramater;  
4. Number of repeated experiments from random initialization with same MathONet structure and regularization parameter. 

## Evaluation

To evaluate my model

```eval
python eval.py
```

## Identified Optimal MathONet Models in the paper:
You can find the identified MathONet models for each dynamic system in this folder. 
