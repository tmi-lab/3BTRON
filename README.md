# 3BTRON: A Blood-Brain Barrier Recognition Network

[![DOI](https://zenodo.org/badge/930454393.svg)](https://doi.org/10.5281/zenodo.15682541)

This is the repository associated with the paper "[3BTRON: A Blood-Brain Barrier Recognition Network](https://doi.org/10.1038/s42003-025-08453-6)". 

## Files

Here, we provide a description of the files made available in this repository.

### The Model

To promote the sharing of resources, we provide the trained model described in the paper, optimised for the PyTorch framework (3BTRON.pt).

### Scripts

This folder contains all associated code (including scripts for data pre-processing, training and evaluation, stratification, and for generating and visualising GradCAM heatmaps).

### Data

The data presented in this study can be found on Zenodo at: https://doi.org/10.5281/zenodo.14845497.

### Experiments

Code for experiments and figures presented in this study will be made available by the corresponding author upon reasonable request. 

## Set-up

For this, you will need to have conda installed (find more information here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Before creating the environment, you will need to update your base environment. If you don't do this, you will receive a numpy error when trying to load and run the model. Update your base environment in your terminal:
```python
conda update --all
```

Now, still in the terminal, you can create the environment from the environment.yml file:
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

## Loading the Model

To generate outputs on your own data, you will first need to load the model before setting the model to evaluation mode. In a new Jupyter notebook, run the following:
```python
import torch
from torchvision import models
from scripts.model import mixedresnetnetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50 = models.resnet50(weights='DEFAULT')
model = mixedresnetnetwork(model=resnet50, embeddings=resnet50.fc.in_features)

SAVE_END_MODEL=True

if SAVE_END_MODEL:
	# for this to work, your notebook must be saved in the same folder as '3BTRON.pt' and the 'scripts' folder.
	# depending on your machine, comment out the line you don't need
    ## if using a gpu
    model.load_state_dict(torch.load('./3BTRON.pt')) 
    ## if running on a CPU-only machine
    model.load_state_dict(torch.load('./3BTRON.pt', map_location=torch.device('cpu'))) 

model = model.to(device)
model.eval()
```
To run the model on your own unlabelled data, you can use the notebook provided above ('generate_outputs.ipynb'). 

This repository includes the preprocessing code ('preprocessing.py', specifically the 'middle_age_data_preprocessing' class for unlabelled data) designed for the datasets used in our study. The preprocessing assumes certain file naming conventions and metadata formats (e.g. CSV column names like path, label, sex, and region). If your dataset differs in structure, naming, or formatting, you can still reproduce our pipeline by adapting your data to meet the following assumptions:

- Each image file name should uniquely identify a sample (e.g. sampleIDvesselID_extension.tif).
- The CSV file should contain at least the following columns:
 - path: matching the image identifiers (before extension/underscores)
 - label: chronological age group (e.g. "Middle age")
 - sex: categorical value ('F' or 'M')
 - region: categorical value ('CC', 'HC', or 'PFC')
- All images should be readable by the load_image() function and stored under data_dir.
  
If your dataset uses different conventions, you can modify the preprocessing accordingly, for instance, by:
- Updating how paths are parsed in the section: images.loc[i, 'path'] = images.loc[i, 'path'].split("/")[-1].split("_")[0].split(".")[0]
- Renaming columns in your metadata CSV to match expected column names.
- Adjusting possible_sexes or possible_regions if your dataset includes different categories.

We encourage users to document any modifications made to preprocessing to ensure transparency and reproducibility.

To fine-tune the model on your own data, you will need to set the model to training mode using:
```python
model.train()
```
To adapt the model to your own data, you will need to retrain the model from scratch.

## Citation

If you use this code in any way, please refer to it by citing my paper "[3BTRON: A Blood-Brain Barrier Recognition Network](https://doi.org/10.1038/s42003-025-08453-6)": 

- Bibtex:
```
@article{Fletcher-Lloyd,
	author={Nan Fletcher-Lloyd and Isabel Bravo-Ferrer and Katrine Gaasdal-Bech and Blanca Díaz Castro and Payam Barnaghi},
	year={2025},
    month={Jul 4},
	title={{3BTRON}: A Blood-Brain Barrier Recognition Network},
	journal={Communications Biology},
	volume={8},
	number={1},
	pages={1001},
	isbn={2399-3642},
	url={https://link.springer.com/article/10.1038/s42003-025-08453-6},
	doi={10.1038/s42003-025-08453-6},
	pmid={40615521}
}
```
## Contact

This code in maintained by Nan Fletcher-Lloyd. 
