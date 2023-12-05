# cpsc583-project-code

## Introduction

This is an implementation of two different models on two different datasets using `PyTorch` and `PyTorch-Geometric`. 

## Usage

First you need to create and enter the corresponding conda environment using the following code.

```{bash}
conda env create -f environment.yml
conda activate pyg
```

Then you can perform training on each dataset with different models using the following code.

```{bash}
# Available models: MolGAN, MolVAE
# Available datasets: QM9, ZINC
python <Model name>_<Dataset name>_training.py
```

You can generate new molecules with a trained model using the following code.

```{bash}
python <Model name>_<Dataset name>_evaluation.py
```
