# Importing the necessary libraries:

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from scipy import stats
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio 

import torch
#import torch.nn as nn

from mbfdunetln.model.mbfdunetln import MBPFDUNet
from mbfdunetln.train import createForwMat, predict_out 
from data.dasandubp import DAS, SensorMaskCartCircleArc

# ---------------------------------------------------------------------------
def gettestdata(cache_dir,ntest):
    
    print('Obtaining data for testing...')
    
    #cache_dir = '../data/cache'

    X0 = np.load(os.path.join(cache_dir, 'Xtest0.npy')) # Noisy sinogram
    X1 = np.load(os.path.join(cache_dir, 'Xtest1.npy'))
    X = np.append(X0,X1,axis=0); 
    #X = X0 # only data with high uncertainty
    #X = X1 # only data with low uncertainty
    del X0,X1
    Y0 = np.load(os.path.join(cache_dir, 'Ytest0.npy')) # True image
    Y1 = np.load(os.path.join(cache_dir, 'Ytest1.npy'))
    Y = np.append(Y0,Y1,axis=0);
    #Y = Y0 # only data with high uncertainty
    #Y = Y1 # only data with low uncertainty
    del Y0,Y1
    #Z0 = np.load(os.path.join(cache_dir, 'Ztest0.npy')) # True sinogram
    #Z1 = np.load(os.path.join(cache_dir, 'Ztest1.npy'))
    #Z = np.append(Z0,Z1,axis=0);
    #Z = Z0 # only data with high uncertainty
    #del Z0,Z1
    
    # Shuffle data
    indpat = np.arange(0, X.shape[0], dtype=int)  # indice patrón
    ida = np.random.permutation(indpat)
    X = X[ida,:,:]
    Y = Y[ida,:,:]
    #Z = Z[ida,:,:]
    
    X = X[0:ntest,:,:]
    Y = Y[0:ntest,:,:]
    #Z = Z[0:ntest,:,:]
    
    X=X.astype(np.float32)
    Y=Y.astype(np.float32)
    #Z=Z.astype(np.float32)
    print('done')
    
    return X,Y #,Z

# ---------------------------------------------------------------------------
def applyDAS(p): # 
    # Experimental setup parameters
    Ns=32      # number of detectors
    Nt=512      # number of time samples
    dx=100e-6   # pixel size  in the x direction [m] 
    nx=64       # number of pixels in the x direction for a 2-D image region
    dsa=8.35e-3 # radius of the circunference where the detectors are placed [m]
    arco=360
    vs=1479;    # speed of sound [m/s]
    to=2e-6        # initial time [s]
    tf=15e-6    # final time [s]      
    t = np.linspace(to, tf, Nt) # time grid
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    Pdas = DAS(nx,dx,dsa,posSens,vs,t,p)
    
    return Pdas

# ---------------------------------------------------------------------------
def predict_mbfdunetln(ntest,fecha):
    
    device = "cpu"
    cache_dir = '../data/cache/'
    
    # Loading test data
    X,Y = gettestdata(cache_dir,ntest)
    
    print('Loading network...')
    net = MBPFDUNet()
    ckp_best='mbfdunetln_best' + fecha + '.pth'  # name of the file of the saved weights of the trained net
    checkpoint = torch.load(ckp_best,map_location=torch.device(device))
    net.load_state_dict(checkpoint['state_dict'])
    print('done')
    
    print('Predicting...')
    Ao = createForwMat()
    pred = predict_out(net, X, Ao, device)
    
    SSIM=np.zeros((X.shape[0],3))
    PC=np.zeros((X.shape[0],3))
    RMSE=np.zeros((X.shape[0],3))
    PSNR=np.zeros((X.shape[0],3))
    print('done')
    
    print('Calculating metrics and doing comparison with DAS...')
    for i1 in range(0,X.shape[0]):
        trueimage=Y[i1,:,:].astype(np.float32);
        predimage=pred[i1,:,:].astype(np.float32); predimage=predimage/np.max(np.abs(predimage));
        SSIM[i1,0]=structural_similarity(trueimage,predimage) 
        PC[i1,0]=stats.pearsonr(trueimage.ravel(),predimage.ravel())[0]  
        RMSE[i1,0]=math.sqrt(mean_squared_error(trueimage,predimage))
        PSNR[i1,0]=peak_signal_noise_ratio(trueimage,predimage)
        Plbp = Ao.T@X[i1,:,:].ravel(); Plbp=Plbp/np.max(np.abs(Plbp)); Plbp=np.reshape(Plbp,(64,64)); Plbp=Plbp.astype(np.float32)
        SSIM[i1,1]=structural_similarity(trueimage,Plbp) 
        PC[i1,1]=stats.pearsonr(trueimage.ravel(),Plbp.ravel())[0]  
        RMSE[i1,1]=math.sqrt(mean_squared_error(trueimage,Plbp))
        PSNR[i1,1]=peak_signal_noise_ratio(trueimage,Plbp)
        Pdas = applyDAS(X[i1,:,:]); Pdas=Pdas/np.max(np.abs(Pdas)); Pdas=np.reshape(Pdas,(64,64)); Pdas=Pdas.astype(np.float32)
        SSIM[i1,2]=structural_similarity(trueimage,Pdas) 
        PC[i1,2]=stats.pearsonr(trueimage.ravel(),Pdas.ravel())[0]  
        RMSE[i1,2]=math.sqrt(mean_squared_error(trueimage,Pdas))
        PSNR[i1,2]=peak_signal_noise_ratio(trueimage,Pdas)
    
    print('\n')
    print('############################################################### \n')
    print('Metrics results NET: \n', 'SSIM: ',round(np.mean(SSIM[:,0]),3), ' PC: ', round(np.mean(PC[:,0]),3), ' RMSE: ', round(np.mean(RMSE[:,0]),3), ' PSNR: ', round(np.mean(PSNR[:,0]),3))
    print('Metrics results LBP: \n', 'SSIM: ',round(np.mean(SSIM[:,1]),3), ' PC: ', round(np.mean(PC[:,1]),3), ' RMSE: ', round(np.mean(RMSE[:,1]),3), ' PSNR: ', round(np.mean(PSNR[:,1]),3))
    print('Metrics results DAS: \n', 'SSIM: ',round(np.mean(SSIM[:,2]),3), ' PC: ', round(np.mean(PC[:,2]),3), ' RMSE: ', round(np.mean(RMSE[:,2]),3), ' PSNR: ', round(np.mean(PSNR[:,2]),3))
    print('\n')
    print('############################################################### \n')
    
    nx = 64; 
    Dx = 100e-6;
    tim = nx*Dx
    colormap=plt.cm.gist_heat
    #colormap=plt.cm.gray
    plt.figure();
    plt.grid(False)
    plt.subplot(1,4,1);plt.xlabel('x (mm)'); plt.ylabel('y (mm)'); plt.title('True image',fontsize=8);
    plt.imshow(trueimage, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
    plt.subplot(1,4,2);plt.xlabel('x (mm)'); plt.title('DAS reconstruction',fontsize=8);
    plt.imshow(Pdas, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,4,3);plt.xlabel('x (mm)');  plt.title('LBP reconstruction',fontsize=8);
    plt.imshow(Plbp, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);  
    plt.subplot(1,4,4);plt.xlabel('x (mm)'); plt.title('predicted image',fontsize=8);
    plt.imshow(predimage, aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);    
    
    return SSIM,PC,RMSE,PSNR