import os
import torch

def load_model(network_name, device, model, optimizer=None, train_mode=True, load_idx=-1):
    weight_folder = os.path.join("weights", network_name)
    os.makedirs(weight_folder, exist_ok=True)
    try:
        weight_list = os.listdir(f'{weight_folder}')
        # latest_pkl_path = sorted(weight_list)[-1]
        weight_list.sort(key= lambda x: int(x.strip('.pkl'))) # invert string to int for correct ordering
        latest_pkl_path = weight_list[load_idx]
        checkpoint = torch.load(f'{weight_folder}/{latest_pkl_path}',  map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if train_mode:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_load = checkpoint['epoch']
        print(f'load the {epoch_load}th epoch from {weight_folder}')
    except:
        print(f'no model found for {network_name}')
        epoch_load = 0
    if train_mode:
        return model, optimizer, epoch_load
    else:
        return model, epoch_load

def save_model(network_name, model, optimizer, epoch):
    weight_folder = os.path.join("weights", network_name)
    os.makedirs(weight_folder, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, f"{weight_folder}/{epoch}.pkl")

def load_model_lr(network_name, device, model, optimizer=None, scheduler=None, train_mode=True, load_idx=-1):
    weight_folder = os.path.join("weights", network_name)
    os.makedirs(weight_folder, exist_ok=True)
    try:
        weight_list = os.listdir(f'{weight_folder}')
        # latest_pkl_path = sorted(weight_list)[-1]
        weight_list.sort(key= lambda x: int(x.strip('.pkl'))) # invert string to int for correct ordering
        latest_pkl_path = weight_list[load_idx]
        checkpoint = torch.load(f'{weight_folder}/{latest_pkl_path}',  map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if train_mode:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                print("no scheduler")
        epoch_load = checkpoint['epoch']
        print(f'load the {epoch_load}th epoch from {weight_folder}')
    except:
        print(f'no model found for {network_name}')
        epoch_load = 0
    if train_mode:
        return model, optimizer, scheduler, epoch_load
    else:
        return model, epoch_load

def save_model_lr(network_name, model, optimizer, scheduler, epoch):
    weight_folder = os.path.join("weights", network_name)
    os.makedirs(weight_folder, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, f"{weight_folder}/{epoch}.pkl")