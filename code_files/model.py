import torch
import torch.nn as nn

class mixedresnetnetwork(nn.Module):
    def __init__(self, model, embeddings, dropout_rate=0.5, num_classes=2, layers_to_unfreeze=None):

        super(mixedresnetnetwork, self).__init__()

        self.image_features_ = model
        for param in self.image_features_.parameters():
            param.requires_grad = False
        if layers_to_unfreeze is not None:
            for name, param in self.image_features_.named_parameters():
                if any(layer in name for layer in layers_to_unfreeze):
                    param.requires_grad = True
        self.image_features_.fc = nn.Identity()

        self.combined_features_ = nn.Sequential(
                nn.Linear((embeddings+5), 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        self.numeric_features_ = None # To store the numeric features captured by the hook

    def forward(self, data):
        split = torch.split(data, [(data.size(2)-5), 5], dim=2)
        images = split[0].view(split[0].shape[0], 3, 224, 224)
        descriptors = split[1]
        x = self.image_features_(images)
        self.numeric_features_ = descriptors # Capture numeric features before further processing; this is where we will store the numeric features using a hook 
        y = self.numeric_features_
        z = torch.cat((x.view(x.size(0), -1), y.view(y.size(0), -1)), dim=1)
        z = self.combined_features_(z)
        return z

    def get_numeric_features(self):
        return self.numeric_features_ # Accessor method to get the numeric features
