import torch
import numpy as np


def get_data_for_mia_base(model, dataloaders, device):
    Y = []
    X = []
    model.to(device)
    model.eval()
    for phase in ['train', 'test']:
        for batch_idx, (data, target) in enumerate(dataloaders[phase]):
            with torch.no_grad():
                inputs, labels = data.to(device), target.to(device)
                output = torch.softmax(model(inputs), dim=1)
            for out in output.cpu().detach().numpy():
                X.append(out)
                if phase == "train":
                    Y.append(1)
                else:
                    Y.append(0)
    return (np.array(X), np.array(Y))


def get_data_for_attack_eval(model, dataloader, device):
    label = []
    pred = []
    soft_pred = []
    model.to(device)
    model.eval()
    for batch_idx, (data, target) in enumerate(dataloader):
        with torch.no_grad():
            inputs, labels = data.to(device), target.to(device)
            output = torch.softmax(model(inputs), dim=1)
            preds = torch.argmax(output, dim=1)
            soft_preds = output[:, 1]
        for cla in labels.cpu().detach().numpy():
            label.append(cla)
        for out in preds.cpu().detach().numpy():
            pred.append(out)
        for s_out in soft_preds.cpu().detach().numpy():
            soft_pred.append(s_out)    
    return (np.array(label), np.array(pred), np.array(soft_pred))


def train_test_acc(model, dataloaders, dataset_sizes, device):
    model.to(device)
    model.eval()
    acc = []
    for phase in ['train', 'test']:
        running_corrects = 0
        for batch_idx, (data, target) in enumerate( dataloaders[phase]):
            with torch.no_grad():
                inputs, labels = data.to(device), target.to(device)
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
        accuracy = running_corrects.double() / dataset_sizes[phase]
        acc.append(accuracy.cpu().detach().numpy())
    return np.array(acc)
