import torch
import numpy as np
import os
from utils.causal_utils import predicting


def load_traindataset_nc(cache_dir,val_percent,train_batchsize,val_batchsize,le):
    train_loaders = []
    val_set = []
    dataset_X = np.load(os.path.join(cache_dir, 'X0.npy'))
    dataset_Y = np.load(os.path.join(cache_dir, 'Y0.npy'))
    for i in range(1,le):
        dataset_X = np.append(dataset_X, np.load(os.path.join(cache_dir, 'X'+str(i)+'.npy')), axis=0)
        dataset_Y = np.append(dataset_Y, np.load(os.path.join(cache_dir, 'Y'+str(i)+'.npy')), axis=0)
    # Shuffle data
    ida = np.random.permutation(np.arange(0, dataset_X.shape[0], dtype=int))
    dataset_X = dataset_X[ida,:,:]
    dataset_Y = dataset_Y[ida,:,:]   
    train_set = torch.utils.data.TensorDataset(torch.tensor(dataset_X).float(), torch.tensor(dataset_Y).float())
    del dataset_X,dataset_Y
    n_val = int(len(train_set) * val_percent)
    n_train = len(train_set) - n_val
    train_set, val_set = torch.utils.data.random_split(train_set, [n_train, n_val])
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=train_batchsize, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=val_batchsize, shuffle=True, drop_last=True)
    return train_loader, val_loader


def train_nc(model,device,train_loader,optimizer,Ao,loss_fn,scheduler):

    model.train()
    losses = []
    example_count = 0
    batch_idx = 0
    
    iterator = iter(train_loader)
    while 1:
        try:
            datas = next(iterator)
        except StopIteration:
            break
        optimizer.zero_grad()       
        predt = predicting(model, datas[0].to(device), Ao, device)
        mean_loss = loss_fn(predt, datas[1].to(device))
        mean_loss.backward()
        optimizer.step()

        losses.append(mean_loss.item())
        example_count += output.shape[0]
        batch_idx += 1
    scheduler.step()    

