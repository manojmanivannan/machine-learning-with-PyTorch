import numpy as np
from PIL import Image


import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import CyclicLR

from prediction_models.genericRegressionClassification import StepByStep, CNN2
from data_generation.rps import download_rps
from plots.stage5 import *

download_rps(localfolder='dataset/RPS')
fig = figure1(folder='dataset/RPS/rps');plt.show()

# Loads temporary dataset to build normalizer
temp_transform = Compose([Resize(28), ToTensor()])
temp_dataset = ImageFolder(root='dataset/RPS/rps', transform=temp_transform)

# Loads temporary dataset to build normalizer
temp_loader = DataLoader(temp_dataset, batch_size=16)
normalizer = StepByStep.make_normalizer(temp_loader)

# Builds transformation, datasets and data loaders
composer = Compose([Resize(28),
                    ToTensor(),
                    normalizer])

train_data = ImageFolder(root='dataset/RPS/rps', transform=composer)
val_data = ImageFolder(root='dataset/RPS/rps-test-set', transform=composer)

# Builds a loader of each set
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

torch.manual_seed(13)
model_cnn3 = CNN2(n_feature=5, p=0.5)
multi_loss_fn = nn.CrossEntropyLoss(reduction='mean')
optimizer_cnn3 = optim.SGD(model_cnn3.parameters(), lr=1e-3, momentum=0.9, nesterov=True)

sbs_cnn3 = StepByStep(model_cnn3, multi_loss_fn, optimizer_cnn3)
tracking, fig = sbs_cnn3.lr_range_test(train_loader, end_lr=2e-1, num_iter=100); plt.show()


optimizer_cnn3 = optim.SGD(model_cnn3.parameters(), lr=0.01, momentum=0.9, nesterov=True)
sbs_cnn3.set_optimizer(optimizer_cnn3)

scheduler = CyclicLR(optimizer_cnn3, base_lr=1e-3, max_lr=0.01, step_size_up=len(train_loader), mode='triangular2')
sbs_cnn3.set_lr_scheduler(scheduler)

sbs_cnn3.set_loaders(train_loader, val_loader)
sbs_cnn3.train(10)

fig = sbs_cnn3.plot_losses(); plt.show()

print(StepByStep.loader_apply(train_loader, sbs_cnn3.correct).sum(axis=0), 
      StepByStep.loader_apply(val_loader, sbs_cnn3.correct).sum(axis=0))


