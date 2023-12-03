import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Normalize, Compose

# Pre-processing the images
# timer
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
        DataLoader(image_datasets['train'], 32,True, num_workers=0),
    'test':
        DataLoader(image_datasets['test'], 32, True, num_workers=0)
}

if torch.cuda.is_available():
    print("Cuda is available")
else:
    print("CPU is available")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize resnet model with pretrained weights
model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)

# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Add a dense layer
model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 525)
).to(device)

# define loss calculator and custom layer optimiser
lossFn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# define the model training fucntion
def train_model(model, optimizer, lossFn, epochCnt):

    for epoch in range(epochCnt):
        since = time.time()
        print('Epoch: ' + str(epoch+1) + '/' + str(epochCnt))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            runningLoss, correct = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = lossFn(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                runningLoss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)

            epochLoss = runningLoss / len(image_datasets[phase])
            epochAcc = correct.double() / len(image_datasets[phase])

            timeElapsed = time.time() - since
            print(phase + ' loss: ' + str(epochLoss) + '   acc: ' + str(epochAcc.item()) + 'time elapsed(s): ' + str(timeElapsed))

    return model

# train the custom layer, preserving the resnet-50 backbone
modelTrained = train_model(model, optimizer, lossFn, 30)

# save the model weights
torch.save(modelTrained.state_dict(), 'importedweights.pt')

##################################################################################################################################################################

