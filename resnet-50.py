import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Normalize, Compose
from pathlib import Path

# Pre-processing the images
normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
data_transforms = {
    'train':
    Compose([
        transforms.Resize((244, 244)),
        transforms.RandomAffine(90, shear=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        normalize
    ]),
    'test':
    Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        normalize
    ])
}
image_datasets = {
    'train':
        ImageFolder('Bird_Dataset/train', data_transforms['train']),
    'test':
        ImageFolder('Bird_Dataset/test', data_transforms['test'])
}
dataloaders = {
    'train':
        DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=0),
    'test':
        DataLoader(image_datasets['test'], batch_size=32, shuffle=True, num_workers=0)
}

if torch.cuda.is_available():
    print("Cuda is available")
else:
    print("CPU is available")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize resnet model with/without pretrained weights
model = resnet50(weights=True).to(device)

# Freeze all pretrained layers
for param in model.parameters():
    param.requires_grad = True

# Add a dense layer
model.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 525)
).to(device)

# define loss calculator and custom layer optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())


# define the model training fucntion
def train_model(model, optimizer, criterion, epochs):

    for epoch in range(epochs):
        print('Epoch: ' + str(epoch+1) + '/' + str(epochs))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(phase + ' loss/acc: ' + str(epoch_loss) + '/' + str(epoch_acc))

    return model

# train the custom layer, preserving the resnet-50 backbone
trained_model = train_model(model, optimizer, criterion, 3)

# save the model weights
torch.save(trained_model.state_dict(), 'weights2')

##################################################################################################################################################################
