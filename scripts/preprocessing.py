from PIL import Image
import os
import glob
import skimage as ski

import torch
import torch.utils
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd

# functions for image preprocessing

def load_image(path):
    image = Image.open(path)
    # Images can be in one of several different modes.
    # Convert to single consistent mode.
    image = image.convert("L")
    return image

def eq_hist(images):
    images_eq_hist = [
        Image.fromarray(
            ski.exposure.equalize_hist(ski.img_as_float(image))
        )
        for image in images
    ]
    return images_eq_hist

def thresholded(images_eq_hist):
    images_eq_hist_threshold = [
        Image.fromarray(
            ski.img_as_float(image) > ski.filters.threshold_li(ski.img_as_float(image))
        )
        for image in images_eq_hist
    ]
    return images_eq_hist_threshold

def rescaled(images_eq_hist_threshold, factor):
    factor = factor
    images_eq_hist_threshold_rescaled = [
        Image.fromarray(
            ski.transform.rescale(ski.img_as_float(image), factor)
        )
        for image in images_eq_hist_threshold
    ]
    return images_eq_hist_threshold_rescaled

class data_preprocessing():                     # pre-processing labelled data
    def __init__(self, csv_file, data_dir):

        self.csv_file = pd.read_csv(csv_file)
        self.data_dir = data_dir

        images = []
        for path in glob.glob(self.data_dir):
            image = load_image(path)
            images.append({
                'path': path,
                'img': image,
            })
        images = pd.DataFrame(images)
        for i in range(0,len(images)):
            images.loc[i, 'path'] = images.loc[i, 'path'].split("/")[-1].split("_")[0].split(".")[0]
        images_and_labels = images.merge(self.csv_file, on="path", how='left', validate="1:1")
        images_and_labels = images_and_labels.loc[images_and_labels['label'] != 'Middle age'] # specific to our dataset, for binary classification
        images_eq_hist_threshold_rescaled = rescaled(thresholded(eq_hist(images_and_labels['img'])), 0.125)
        np_images = []
        for image in range(len(images_eq_hist_threshold_rescaled)):
            pixels = ski.img_as_float(images_eq_hist_threshold_rescaled[image])
            np_images.append(pixels)
        np_images = np.array(np_images)
        
        categories = {
            'Young': 0,
            'Aged': 1,
            } 
        categories_num = images_and_labels['label'].map(categories)
        labels = categories_num.values

        features = images_and_labels[['sex','region']]
        onehot_features = pd.get_dummies(features, columns = ['sex', 'region'], dtype='float32')
        onehot_features = np.array(onehot_features)

        folder_path = os.path.dirname(path)

        dataset_folder = os.path.join(folder_path, 'binary') 
        for folder_path in [dataset_folder]:
             if not os.path.exists(folder_path):
                  os.makedirs(folder_path)
                  
        np.save(os.path.join(dataset_folder, 'np_images.npy'), np_images)
        np.save(os.path.join(dataset_folder, 'labels.npy'), labels)
        np.save(os.path.join(dataset_folder, 'features.npy'), onehot_features)

class middle_age_data_preprocessing():          # pre-processing unlabelled data
    def __init__(self, csv_file, data_dir):

        self.csv_file = pd.read_csv(csv_file)
        self.data_dir = data_dir

        images = []
        for path in glob.glob(self.data_dir):
            image = load_image(path)
            images.append({
                'path': path,
                'img': image,
            })
        images = pd.DataFrame(images)
        for i in range(0,len(images)):
            images.loc[i, 'path'] = images.loc[i, 'path'].split("/")[-1].split("_")[0].split(".")[0]
        images_and_labels = images.merge(self.csv_file, on="path", how='left', validate="1:1")
        images_and_labels = images_and_labels.loc[images_and_labels['label'] == 'Middle age'] # specific to our dataset, to investigate middle age mouse brains
        images_eq_hist_threshold_rescaled = rescaled(thresholded(eq_hist(images_and_labels['img'])), 0.125)
        np_images = []
        for image in range(len(images_eq_hist_threshold_rescaled)):
            pixels = ski.img_as_float(images_eq_hist_threshold_rescaled[image])
            np_images.append(pixels)
        np_images = np.array(np_images)

        features = images_and_labels[['sex','region']]

        possible_sexes = ['F', 'M']  # specific to our dataset, defines the 2 possible sexes 
        possible_regions = ['CC', 'HC', 'PFC']  # specific to our datasets, defines the 3 possible regions

        features['sex'] = pd.Categorical(features['sex'], categories=possible_sexes) # specific to our dataset
        features['region'] = pd.Categorical(features['region'], categories=possible_regions) # specific to our dataset
        
        onehot_features = pd.get_dummies(features, columns = ['sex', 'region'], dtype='float32')
        onehot_features = np.array(onehot_features)

        folder_path = os.path.dirname(path)

        dataset_folder = os.path.join(folder_path, 'middle_age') 
        for folder_path in [dataset_folder]:
             if not os.path.exists(folder_path):
                  os.makedirs(folder_path)
                  
        np.save(os.path.join(dataset_folder, 'np_images.npy'), np_images)
        np.save(os.path.join(dataset_folder, 'features.npy'), onehot_features)

class train_test_split_func():
    def __init__(self, data_dir):
        
        self.data_dir = data_dir

        np_images = np.load(self.data_dir+'/np_images.npy')
        labels = np.load(self.data_dir+'/labels.npy')
        features = np.load(self.data_dir+'/features.npy')

        x_images_train, x_images_test, x_features_train, x_features_test, y_train, y_test = train_test_split(np_images, features, labels, test_size=0.10, random_state=44, stratify=labels)

        train_folder = os.path.join(self.data_dir, 'train')
        test_folder = os.path.join(self.data_dir, 'test')
        
        for folder_path in [train_folder, test_folder]:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        np.save(os.path.join(train_folder, 'np_images.npy'), x_images_train)
        np.save(os.path.join(train_folder, 'features.npy'), x_features_train)
        np.save(os.path.join(train_folder, 'labels.npy'), y_train)
        np.save(os.path.join(test_folder, 'np_images.npy'), x_images_test)
        np.save(os.path.join(test_folder, 'features.npy'), x_features_test)
        np.save(os.path.join(test_folder, 'labels.npy'), y_test)

class get_dataset(Dataset):             # for labelled data
    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir

        np_images = np.load(self.data_dir+'/np_images.npy')
        labels = np.load(self.data_dir+'/labels.npy')
        features = np.load(self.data_dir+'/features.npy')

        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        self.sample_weights = np.array([class_weights[label] for label in labels])

        images_tensor, features_tensor, labels_tensor = torch.tensor(np_images), torch.tensor(features), torch.tensor(labels)
        images_tensor = torch.unsqueeze(images_tensor, 1)
        tensors = (images_tensor, features_tensor, labels_tensor)

        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

        self.tensors = tensors

        self.transform = transform

    def __getitem__(self, idx):

        trans_x1 = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1) if x.size(0)==1 else x)
        trans_x2 = transforms.Lambda(lambda x: x.repeat(1, 1))

        x_1 = self.tensors[0][idx]
        x_1 = trans_x1(x_1)

        if self.transform is not None:
            x_1 = self.transform(x_1)

        x_2 = self.tensors[1][idx]
        x_2 = trans_x2(x_2)

        x = torch.cat((x_1.view(x_1.size(0), -1), x_2.view(x_2.size(0), -1)), dim=1)
        y = self.tensors[2][idx]
        sample_weight = self.sample_weights[idx]
        
        return x, y, sample_weight

    def __len__(self):
        return self.tensors[0].size(0)

class get_middle_age_dataset(Dataset):      # for unlabelled data
    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir

        np_images = np.load(self.data_dir+'/np_images.npy')
        features = np.load(self.data_dir+'/features.npy')

        images_tensor, features_tensor = torch.tensor(np_images), torch.tensor(features)
        images_tensor = torch.unsqueeze(images_tensor, 1)
        tensors = (images_tensor, features_tensor)

        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

        self.tensors = tensors

        self.transform = transform

    def __getitem__(self, idx):

        trans_x1 = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1) if x.size(0)==1 else x)
        trans_x2 = transforms.Lambda(lambda x: x.repeat(1, 1))

        x_1 = self.tensors[0][idx]
        x_1 = trans_x1(x_1)

        if self.transform is not None:
            x_1 = self.transform(x_1)

        x_2 = self.tensors[1][idx]
        x_2 = trans_x2(x_2)

        x = torch.cat((x_1.view(x_1.size(0), -1), x_2.view(x_2.size(0), -1)), dim=1)
        
        return x

    def __len__(self):
        return self.tensors[0].size(0)

class gridsearch_split(Dataset):
    def __init__(self, images, features, targets, transform=None):

        self.images = images
        self.features = features
        self.targets = targets

        class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
        self.sample_weights = np.array([class_weights[label] for label in targets])

        images_tensor, features_tensor, labels_tensor = torch.tensor(self.images), torch.tensor(self.features), torch.tensor(self.targets)
        images_tensor = torch.unsqueeze(images_tensor, 1)
        tensors = (images_tensor, features_tensor, labels_tensor)

        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

        self.tensors = tensors

        self.transform = transform

    def __getitem__(self, idx):

        trans_x1 = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1) if x.size(0)==1 else x)
        trans_x2 = transforms.Lambda(lambda x: x.repeat(1, 1))

        x_1 = self.tensors[0][idx]
        x_1 = trans_x1(x_1)

        if self.transform is not None:
            x_1 = self.transform(x_1)

        x_2 = self.tensors[1][idx]
        x_2 = trans_x2(x_2)

        x = torch.cat((x_1.view(x_1.size(0), -1), x_2.view(x_2.size(0), -1)), dim=1)
        y = self.tensors[2][idx]
        sample_weight = self.sample_weights[idx]
        
        return x, y, sample_weight

    def __len__(self):
        return self.tensors[0].size(0) 

class to_memory(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, device):
        self.dataset = dataset
        self.memory_dataset = {}
        self.device = device

    def __getitem__(self, index):
        if index in self.memory_dataset:
            return self.memory_dataset[index]
        output = self.dataset[index]
        self.memory_dataset[index] = output
        return output

    def __len__(self):
        return len(self.dataset)
