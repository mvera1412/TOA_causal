from skimage.transform import AffineTransform, warp
import numpy as np
import random


def augmentate_letterdata(MI):


    # Number of images without augmentation
    Ni = int(np.size(MI, 1))
    
    ap = 5
    MI2 = np.zeros((MI.shape[0],ap*Ni))
    cont = -1*ap
    nx = 64
    
    for i0 in range(0,Ni):
        cont = cont + ap
        MI2[:, cont] = MI[:,i0]
        aux = np.reshape(MI[:,i0],(nx,nx))
        MI2[:,cont + 1]=np.reshape(aux[:,::-1],(int(nx**2),)) # Horizontal flip
        #MItraina[:,cont + 2]=np.reshape(aux[::-1,:],(int(nx**2),)) # Vertical flip
        tf = AffineTransform(translation=(random.randint(0,16),random.randint(0,16)))
        MI2[:,cont + 2]=np.reshape(warp(aux,tf,mode="wrap"),(int(nx**2),)) # Translation 
        tf = AffineTransform(shear=0.25)
        MI2[:,cont + 3]=np.reshape(warp(aux,tf,order=1,preserve_range=True,mode='wrap'),(int(nx**2),)) # sheared 1 
        tf = AffineTransform(shear=-0.25)
        MI2[:,cont + 4]=np.reshape(warp(aux,tf,order=1,preserve_range=True,mode='wrap'),(int(nx**2),)) # sheared 2
        
    return MI2