# MathONet

This repository is the implementation of [Bayesian Learning to Discover Mathematical Operations in Governing Equations of Dynamic Systems]

## Description

This paper presents a method that can learn the mathematical operations in governing equations of dynamic systems composed of the basic mathematical operations, i.e., unary and binary operations. The governing equations are formulated as a DenseNet-like hierarchical structure, termed as MathONet. The algorithm is demonstrated on the chaotic Lorenz system, Lotka-Volterra system and Kolmogorov–Petrovsky–Piskunov (Fisher-KPP) system. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Generate data

To generate data for each benchmark run this command in each folder:

```train
python data_generate.py
```

## Training and Evaluation

To train and evaluate the model(s) in the paper, run this command in each folder:

```train
python main.py
```

>📋  Several hyperparameters needs to be defined in the train.py, including: 
1. Number of hidden layers and hidden neurons for MathONet; 
2. Initialization of unary operations.
3. Value of regularization paramater;  
4. Number of repeated experiments from random initialization with same MathONet structure and regularization parameter. 

