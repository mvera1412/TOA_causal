import torch
#import ignite
import numpy as np
import os
import shutil

#from TOA.mbfdunetln import MBPFDUNet
#from ANDMask.adam_flexible_weight_decay import AdamFlexibleWeightDecay
#from torch.optim.lr_scheduler import MultiStepLR
from TOA.predict import applyDAS
import ANDMask.and_mask_utils as and_mask_utils
from ANDMask.common import permutation_groups
import math
from scipy import stats
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio 
import matplotlib.pyplot as plt


def load_traindataset(cache_dir,val_percent,train_batchsize,val_batchsize,le):
    train_loaders = []
    val_set = []
    for i in range(le):
        X = np.load(os.path.join(cache_dir, 'X'+str(i)+'.npy')) #Lotes de 2216
        Y = np.load(os.path.join(cache_dir, 'Y'+str(i)+'.npy'))
        train_set = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(Y).float())
        del X,Y
        n_val = int(len(train_set) * val_percent)
        n_train = len(train_set) - n_val
        train_set, v = torch.utils.data.random_split(train_set, [n_train, n_val])
        val_set.append(v)
        data_loader = torch.utils.data.DataLoader(train_set,batch_size=train_batchsize, shuffle=True, drop_last=True) #1776 datos
        del train_set
        train_loaders.append(data_loader)
    # Son 2200, voy a hacer 55 grupos de 40
    val_set = torch.utils.data.ConcatDataset(val_set)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=val_batchsize, shuffle=True, drop_last=True)
    return train_loaders, val_loader

def load_testdataset(cache_dir):
    # Son 616, voy a hacer 14 grupos de 44
    bs_test = 44
    test_loaders = []
    X0 = np.load(os.path.join(cache_dir, 'Xtest0.npy'))
    Y0 = np.load(os.path.join(cache_dir, 'Ytest0.npy'))
    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X0).float(), torch.tensor(Y0).float()),batch_size=bs_test, shuffle=True, drop_last=True)
    del X0,Y0
    test_loaders.append(data_loader)
    X1 = np.load(os.path.join(cache_dir, 'Xtest1.npy'))
    Y1 = np.load(os.path.join(cache_dir, 'Ytest1.npy'))
    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X1).float(), torch.tensor(Y1).float()),batch_size=bs_test, shuffle=True, drop_last=True)
    del X1,Y1
    test_loaders.append(data_loader)
    return test_loaders

def applyInvMat(x, Ao, dimS, dimI): # [Ao] = (16384,4096)
    x = torch.squeeze(x,1) # (-1,32,512)
    x = torch.reshape(x,(dimS[0],int(dimS[2]*dimS[3]))) # (-1,16384)
    y = torch.matmul(Ao.T,x.T).T # ((4096,16384) @ (16384,-1)).T = (-1,4096)
    y = torch.reshape(y,(dimI[0],dimI[2],dimI[3])) # (-1,64,64)
    y = torch.unsqueeze(y,1) # (-1,1,64,64)  
    return y

def applyForwMat(y, Ao, dimS, dimI):
    y = torch.squeeze(y,1) # (-1,64,64)
    y = torch.reshape(y,(dimI[0],int(dimI[2]*dimI[3]))) # (-1,4096)
    x = torch.matmul(Ao,y.T).T # ((16384,4096) @ (4096,-1)).T = (-1,16384)
    x = torch.reshape(x,(dimS[0],dimS[2],dimS[3])) # (-1,32,512)
    x = torch.unsqueeze(x,1) # (-1,1,32,512)    
    return x

def predicting(net, input, Ao, device):
    x = input.to(device=device)
    x = torch.unsqueeze(x,1)
    x = x.type(torch.float32)
    dimS = x.shape # (-1,1,128,512)
    dimI = (dimS[0],dimS[1],64,64) # (-1,1,64,64)
    f0 = applyInvMat(x,Ao,dimS,dimI) # (-1,1,64,64)
    g1 = applyForwMat(f0,Ao,dimS,dimI) # (-1,1,32,512)
    Dg = g1 - x # (-1,1,128,512)
    Df = applyInvMat(Dg,Ao,dimS,dimI) # (-1,1,64,64)
    pred = net.to(device=device)(f0,Df)
    return torch.squeeze(pred,1)

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss = checkpoint['valid_loss_min'] 
    return model, optimizer, checkpoint['epoch'], loss

def train(model, device, train_loaders, optimizer,
          n_agreement_envs,
          loss_fn,
          Ao,
          agreement_threshold,
          scheduler):

    model.train()

    losses = []
    example_count = 0
    batch_idx = 0

    train_iterators = [iter(loader) for loader in train_loaders]
    it_groups = permutation_groups(train_iterators, n_agreement_envs)

    while 1:
        train_iterator_selection = next(it_groups)
        try:
            datas = [next(iterator) for iterator in train_iterator_selection]
        except StopIteration:
            break

        assert len(datas) == n_agreement_envs

        batch_size = datas[0][0].shape[0]
        assert all(d[0].shape[0] == batch_size for d in datas)

        inputs = [d[0].to(device) for d in datas]
        target = [d[1].to(device) for d in datas]

        inputs = torch.cat(inputs, dim=0)
        target = torch.cat(target, dim=0)

        optimizer.zero_grad()

        output = predicting(model, inputs, Ao, device)
        mean_loss, masks = and_mask_utils.get_grads(
            agreement_threshold,
            batch_size,
            loss_fn, n_agreement_envs,
            params=optimizer.param_groups[0]['params'],
            output=output,
            target=target,
            method='and_mask',
            scale_grad_inverse_sparsity=1,
        )
        optimizer.step()

        losses.append(mean_loss.item())
        example_count += output.shape[0]
        batch_idx += 1

    scheduler.step()

    
def validation(model, device, val_loader, optimizer, loss_fn, Ao, valid_loss_min, epoch, ckp_last, ckp_best):
    val_loss = 0.0
    bs = val_loader.batch_size
    n_val = bs * len(val_loader)
    with torch.no_grad():
        iterator = iter(val_loader)
        while 1:
            try:
                datas = next(iterator)
            except StopIteration:
                break
            predv = predicting(model, datas[0].to(device), Ao, device)
            loss = loss_fn(predv, datas[1].to(device))
            val_loss += bs * loss.item()
    val_loss = val_loss / n_val     

    checkpoint = {
            'epoch': epoch,
            'valid_loss_min': np.min((valid_loss_min,val_loss)),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
                }
    save_ckp(checkpoint, False, ckp_last, ckp_best)
    if val_loss < valid_loss_min:
        valid_loss_min = val_loss
         save_ckp(checkpoint, True, ckp_last, ckp_best)
    return valid_loss_min 

def computing_metrics(X,Y,Ao,model,model_nc):
  bs = X.shape[0]
  pred = predicting(model,X, Ao.to(device="cpu"), "cpu")
  pred_nc = predicting(model_nc,X, Ao.to(device="cpu"), "cpu")
  SSIM=np.zeros((bs,4))
  PC=np.zeros((bs,4))
  RMSE=np.zeros((bs,4))
  PSNR=np.zeros((bs,4))
  for i1 in range(bs):
    trueimage=Y[i1,:,:].detach().numpy()
    predimage=pred[i1,:,:].detach().numpy()
    predimage=predimage/np.max(np.abs(predimage))
    SSIM[i1,0]=structural_similarity(trueimage,predimage) 
    PC[i1,0]=stats.pearsonr(trueimage.ravel(),predimage.ravel())[0]  
    RMSE[i1,0]=math.sqrt(mean_squared_error(trueimage,predimage))
    PSNR[i1,0]=peak_signal_noise_ratio(trueimage,predimage)
    
    predimage=pred_nc[i1,:,:].detach().numpy()
    predimage=predimage/np.max(np.abs(predimage))
    SSIM[i1,1]=structural_similarity(trueimage,predimage) 
    PC[i1,1]=stats.pearsonr(trueimage.ravel(),predimage.ravel())[0]  
    RMSE[i1,1]=math.sqrt(mean_squared_error(trueimage,predimage))
    PSNR[i1,1]=peak_signal_noise_ratio(trueimage,predimage)
    
    Plbp = Ao.T@X[i1,:,:].ravel()
    Plbp = Plbp.detach().numpy()
    Plbp=Plbp/np.max(np.abs(Plbp))
    Plbp=np.reshape(Plbp,(64,64))
    Plbp=Plbp.astype(np.float32)
    SSIM[i1,2]=structural_similarity(trueimage,Plbp) 
    PC[i1,2]=stats.pearsonr(trueimage.ravel(),Plbp.ravel())[0]  
    RMSE[i1,2]=math.sqrt(mean_squared_error(trueimage,Plbp))
    PSNR[i1,2]=peak_signal_noise_ratio(trueimage,Plbp)
    
    Pdas = applyDAS(X[i1,:,:])
    Pdas=Pdas/np.max(np.abs(Pdas))
    Pdas=np.reshape(Pdas,(64,64))
    Pdas=Pdas.astype(np.float32)
    SSIM[i1,3]=structural_similarity(trueimage,Pdas) 
    PC[i1,3]=stats.pearsonr(trueimage.ravel(),Pdas.ravel())[0]  
    RMSE[i1,3]=math.sqrt(mean_squared_error(trueimage,Pdas))
    PSNR[i1,3]=peak_signal_noise_ratio(trueimage,Pdas)
  return SSIM,PC,RMSE,PSNR

def testing(SSIM,PC,RMSE,PSNR,loader,Ao,model,model_nc):
  dim = SSIM.shape
  nx = 64; 
  Dx = 100e-6
  tim = nx*Dx
  for j in range(dim[0]):
    print(f"\n Environment {j} \n")
    print('############################################################### \n')
    print('Metrics results NET (ANDMask): \n', 'SSIM: ',round(np.mean(SSIM[j,:,:,0]),3), ' PC: ', round(np.mean(PC[j,:,:,0]),3), ' RMSE: ', round(np.mean(RMSE[j,:,:,0]),3), ' PSNR: ', round(np.mean(PSNR[j,:,:,0]),3))
    print('Metrics results NET (benchmark): \n', 'SSIM: ',round(np.mean(SSIM[j,:,:,1]),3), ' PC: ', round(np.mean(PC[j,:,:,1]),3), ' RMSE: ', round(np.mean(RMSE[j,:,:,1]),3), ' PSNR: ', round(np.mean(PSNR[j,:,:,1]),3))    
    print('Metrics results LBP: \n', 'SSIM: ',round(np.mean(SSIM[j,:,:,2]),3), ' PC: ', round(np.mean(PC[j,:,:,2]),3), ' RMSE: ', round(np.mean(RMSE[j,:,:,2]),3), ' PSNR: ', round(np.mean(PSNR[j,:,:,2]),3))
    print('Metrics results DAS: \n', 'SSIM: ',round(np.mean(SSIM[j,:,:,3]),3), ' PC: ', round(np.mean(PC[j,:,:,3]),3), ' RMSE: ', round(np.mean(RMSE[j,:,:,3]),3), ' PSNR: ', round(np.mean(PSNR[j,:,:,3]),3))
    print('\n')
    print('############################################################### \n')
    colormap=plt.cm.gist_heat
    plt.figure(figsize=(19.2, 14.4))
    plt.suptitle('Environment'+str(j),y=0.65)
    plt.grid(False)
    plt.subplot(1,5,1); plt.title('True image',fontsize=12);
    plt.subplots_adjust(wspace=0.5)
    dataset = next(iter(loader[j]))
    rnd_idx = np.random.randint(len(dataset[0]))
    sinogram = dataset[0][rnd_idx]
    trueimage = dataset[1][rnd_idx]
    Pdas = applyDAS(sinogram)
    Pdas=Pdas/np.max(np.abs(Pdas))
    Pdas=np.reshape(Pdas,(64,64))
    Pdas=Pdas.astype(np.float32)
    Plbp = Ao.T@sinogram.ravel()
    Plbp = Plbp.detach().numpy() 
    Plbp=Plbp/np.max(np.abs(Plbp))
    Plbp=np.reshape(Plbp,(64,64))
    Plbp=Plbp.astype(np.float32)
    predimage = predicting(model,sinogram.view(1,sinogram.shape[0],sinogram.shape[1]), Ao, "cpu").detach().numpy()
    predimage_nc = predicting(model_nc,sinogram.view(1,sinogram.shape[0],sinogram.shape[1]), Ao, "cpu").detach().numpy()
    
    plt.imshow(trueimage, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
    plt.subplot(1,5,2);plt.title('DAS reconstruction',fontsize=12);
    plt.imshow(Pdas, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,5,3);plt.title('LBP reconstruction',fontsize=12);
    plt.imshow(Plbp, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
    plt.subplot(1,5,4);plt.title('Benchmark recosntruction',fontsize=12);
    plt.imshow(predimage_nc[0,:,:], aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,5,5);plt.title('ANDMask recosntruction',fontsize=12);
    plt.imshow(predimage[0,:,:], aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);    
