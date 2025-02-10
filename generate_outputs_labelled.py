import torch # type: ignore
import torch.utils # type: ignore
from torchvision import models, transforms # type: ignore

import pandas as pd # type: ignore
import numpy as np # type: ignore

from code_files.utils import ( # type: ignore
    set_seed,
    worker_init_fn
) 
from code_files.preprocessing import ( # type: ignore
    get_middle_age_dataset, 
    to_memory
) 
from code_files.models import ( # type: ignore
    mixedresnetnetwork
) 

torch.__version__

set_seed(44)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f'Selected device: {device}')

torch.cuda.is_available()

test_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

test_data_dir = './data/your_data'

np_images = np.load(test_data_dir + '/np_images.npy')
features = np.load(test_data_dir + '/features.npy')
print(f"Shape of np_images: {np_images.shape}")
print(f"Shape of features: {features.shape}")

test_dataset = get_dataset(test_data_dir, transform=test_transform)

test_dataset = to_memory(test_dataset, device)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)

# Load pre-trained model with default weights

resnet50 = models.resnet50(weights='DEFAULT')
model = mixedresnetnetwork(model=resnet50, embeddings=resnet50.fc.in_features)
print(model)

SAVE_END_MODEL=True

if SAVE_END_MODEL:
    model.load_state_dict(torch.load('./3BTRON.pt'))

model = model.to(device)
model.eval()

optimal_thresholds = {}
optimal_thresholds['3BTRON'] = {'green': 0.25,
                                'amber': 0.75}
print(optimal_thresholds)

# Get numeric features and calibrated probabilities

numeric_features_list = []
all_test_logits = []


# Generate outputs on your own data

with torch.no_grad():
    for batch in test_loader:
        data, targets, _ = batch
        targets = targets.to(device)
        logits = model(data.to(device))
        all_test_logits.append(logits)
        numeric_features_list.append(model.get_numeric_features().cpu().numpy())
        labels.append(targets.cpu().numpy())
numeric_features_array = np.concatenate(numeric_features_list, axis=0)
true_labels = np.concatenate(labels, axis=0)
test_logits = torch.cat(all_test_logits, dim=0)
calibrated_probs = torch.sigmoid(test_logits[:, 1]).cpu().numpy()

scores['3BTRON'] = {
        'test_true_labels': true_labels,
        'test_features': numeric_features_array,
        'test_pos_probabilities': calibrated_probs
    }

final_results = evaluate_on_test_set(scores, optimal_thresholds)
print(final_results)

# Assign traffic light labels
traffic_lights = assign_traffic_light(scores['3BTRON']['test_pos_probabilities'], green_threshold=optimal_thresholds['3BTRON']['green'], amber_threshold=optimal_thresholds['3BTRON']['amber'])
print(traffic_lights)
traffic_lights_df = pd.DataFrame(traffic_lights)
# Get true labels
true_labels = scores['3BTRON']['test_true_labels']
print(true_labels)
true_labels_df = pd.DataFrame(true_labels)
# Get calibrated probabilities
calibrated_probs = scores['3BTRON']['test_pos_probabilities']
print(calibrated_probs)
calibrated_probs_df = pd.DataFrame(calibrated_probs)
# Get features 
features = scores['3BTRON']['test_features']
features = features.reshape(len(features), len(features[0]))
print(features)
features_df = pd.DataFrame(features)
# Get deep look 
deep_look = pd.concat([traffic_lights_df, true_labels_df, calibrated_probs_df, features_df], axis=1)
deep_look.columns = ['Traffic Lights', 'True Labels', 'Age Likelihood', 'Female', 'Male', 'CC', 'HC', 'PFC']
deep_look['Sex'] = deep_look['Female'].map(lambda x: 'Female' if x == 1 else 'Male')
deep_look = deep_look.drop(columns=['Female', 'Male'])
def get_brain_region(row):
    regions = ['CC', 'HC', 'PFC']
    active_regions = [region for region in regions if row[region] == 1]
    # If one brain region is active, return that brain region
    if len(active_regions) == 1:
        return active_regions[0]
    # If multiple brain regions are active, you can either choose one or concatenate them
    else:
        return ', '.join(active_regions)  # Concatenate brain regions 
deep_look['Brain Region'] = deep_look.apply(get_brain_region, axis=1)
deep_look = deep_look.drop(columns=['CC', 'HC', 'PFC'])
print(deep_look)
deep_look.to_csv('3BTRON_labelled_deep_look.csv', index=False)
