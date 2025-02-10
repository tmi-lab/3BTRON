# 3BTRON: A Blood-Brain Barrier Recognition Network

This is the repository associated with the paper "3BTRON: A Blood-Brain Barrier Recognition Network". 

## Files

Here, we provide a description of the files made available in this repository.

### The Model

To promote the sharing of resources, we provide the trained model described in the paper, optimised for the PyTorch framework (3BTRON.pt).

### Scripts

This folder contains all associated code (including scripts for data pre-processing, training and evaluation, stratification and for generating and visualising GradCAM heatmaps).

### Data

The data presented in this study can be found on Zenodo at:

### Experiments

Code for experiments and figures presented in this study will be made available by the corresponding author upon reasonable request. 

## Set-up

For this, you will need to have conda installed (find more information here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Create the environment from the environment.yml file:
```python
conda env create -f environment.yml
```

Activate the environment: 
```python
conda activate img-classification
```

Verify that the environment was installed correctly:
```python
conda env list
```

## Running the Model

To generate outputs on your own data, run the python scripts 'generate_outputs_labelled.py' for labelled data or 'generate_outputs_unlabelled.py' for unlabelled data. The model is in evaluation mode in these scripts.

To fine-tune the model on your data, you will need to set the model to training mode using:
```python
model.train()
```

## Contact

This code in maintained by Nan Fletcher-Lloyd. 
