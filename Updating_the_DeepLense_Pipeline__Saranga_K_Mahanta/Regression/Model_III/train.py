import gc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn

from ranger import Ranger
from utils import device, set_seed, train_transforms, test_transforms
from model import Regressor
from dataloader import create_dataloaders

# REMOVED: static imports from config.py
# ADDED: dynamic configuration loader
from utils.config_loader import get_config 

def train_epoch(model, dataloader, criterion, optimizer, scheduler, example_ct):
    model.train()
    train_loss = []

    loop = tqdm(enumerate(dataloader), total=len(dataloader))

    for batch_idx, (img_batch, labels) in loop:
        X = img_batch.to(device)
        y_truth = labels.to(device)
        example_ct += len(img_batch)
        
        # forward prop
        y_pred = model(X)
        y_pred = y_pred.view(-1)
        
        # loss calculation
        loss = criterion(y_pred, y_truth)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # batch loss
        train_loss.append(loss.detach().cpu().numpy())

    return model, np.mean(train_loss), example_ct


def test_epoch(model, dataloader, criterion):
    model.eval()
    losses = []
    y_pred_list = []
    y_truth_list = []

    with torch.no_grad():
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, (img_batch, masses) in loop:
            X = img_batch.to(device)
            y_truth = masses.to(device)
            y_truth_list.append(y_truth.detach().cpu().numpy())

            # forward prop
            y_pred = model(X)
            y_pred = y_pred.view(-1)
            y_pred_list.append(y_pred.detach().cpu().numpy())

            # loss calculation
            loss = criterion(y_pred, y_truth)
            losses.append(loss.detach().cpu().numpy())

    return y_pred_list, y_truth_list, np.mean(losses)


def plot_results(model, dataloader, criterion, epoch):
    y_pred_list, y_truth_list, test_loss = test_epoch(model, dataloader, criterion)
    
    def flatten_list(x):
        flattened_list = []
        for i in x:
            for j in i:
                flattened_list.append(j)
        return flattened_list
    
    y_pred_list_flattened = flatten_list(y_pred_list)
    y_truth_list_flattened = flatten_list(y_truth_list)
    
    plt.figure(figsize=(9,9))
    plt.scatter(y_truth_list_flattened, y_pred_list_flattened)
    plt.xlabel('Observed mass')
    plt.ylabel('Predicted mass')
    plt.draw()
    

# Passed dataloaders and config as arguments to remove global variable dependencies
def fit_model(model, train_loader, val_loader, test_loader, config):
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    save_model = config['training']['save_model']
    checkpoint_path = config['model']['model_path']

    optimizer = Ranger(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader), verbose=False)
    criterion = nn.MSELoss()

    loss_dict = {'train_loss':[], 'val_loss':[]}
    example_ct = 0  
    min_val_loss = 999 

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}:")
        model, train_loss, example_ct = train_epoch(model, train_loader, criterion, optimizer, scheduler, example_ct)
        _, _, val_loss = test_epoch(model, val_loader, criterion)
        
        if save_model:
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print("New lower val loss. Saving model checkpoint!")
                torch.save(model.state_dict(), checkpoint_path)

        print(f'Train loss: {train_loss}, Val loss: {val_loss}')

        loss_dict['train_loss'].append(train_loss)
        loss_dict['val_loss'].append(val_loss)

        if epoch % 3 == 0 or epoch + 1 == epochs:
            plot_results(model, val_loader, criterion, epoch)

    return model, loss_dict
    
    
if __name__ == "__main__":
    # 1. Load the dynamic configuration
    config = get_config()
    
    set_seed(config['training']['seed'])
    
    # 2. Map the config variables to the dataloader
    train_loader, val_loader, test_loader = create_dataloaders(
        config['data']['train_data_path'], 
        config['data']['test_data_path'], 
        train_transforms, test_transforms, 
        config['training']['batch_size']
    )

    model = Regressor().to(device)

    # 3. Map the config variables to the model loading logic
    if config['model']['load_pretrained_model']:
        if device != 'cpu':
            model.load_state_dict(torch.load(config['model']['model_path']))
        else:
            model.load_state_dict(torch.load(config['model']['model_path'], map_location=torch.device('cpu')))

    # 4. Pass the config into the training loop
    model, loss_dict = fit_model(model, train_loader, val_loader, test_loader, config)

    # Plot losses
    plt.figure(figsize=(19,12))
    plt.semilogy(loss_dict['train_loss'], label='Train')
    plt.semilogy(loss_dict['val_loss'], label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE Loss')
    plt.legend()
    plt.title('Loss history')
    plt.savefig('Loss_history.png')
    plt.show()





