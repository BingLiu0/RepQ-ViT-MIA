import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    print("DATASET SIZE", dataset_sizes)
    since = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    retunr_value_train = np.zeros((4,num_epochs))

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (data, target) in enumerate( dataloaders[phase]):
                inputs, labels = data.to(device), target.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                epoch_loss_train = epoch_loss
                epoch_accuracy_train = epoch_acc
                retunr_value_train[0][epoch] = epoch_loss
                retunr_value_train[1][epoch] = epoch_acc
            else:
                epoch_loss_test = epoch_loss
                epoch_accuracy_test = epoch_acc
                retunr_value_train[2][epoch] = epoch_loss
                retunr_value_train[3][epoch] = epoch_acc

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        scheduler.step()
        tqdm.write(f'Epoch {epoch+1}/{num_epochs} - Train_Loss: {epoch_loss_train} - Test_Loss: {epoch_loss_test}- Train_Accuracy: {epoch_accuracy_train} - Test_Accuracy: {epoch_accuracy_test}')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("DONE TRAIN")

    # model.load_state_dict(best_model_wts)

    return model, retunr_value_train

