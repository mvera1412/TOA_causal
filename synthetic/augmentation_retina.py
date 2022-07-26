#from skimage.transform import AffineTransform, warp
from skimage import util
import numpy as np
#import random


def augmentate_retina(MI):


    # Number of images without augmentation
    Ni = int(np.size(MI, 1))
    
    ap = 4
    MI2 = np.zeros((MI.shape[0],ap*Ni))
    cont = -1*ap
    nx = 64
    
    for i0 in range(0,Ni):
        cont = cont + ap
        MI2[:, cont] = MI[:,i0]
        aux = np.reshape(MI[:,i0],(nx,nx))
        MI2[:,cont + 1]=np.reshape(aux[:,::-1],(int(nx**2),)) # Horizontal flip
        MI2[:,cont + 2]=np.reshape(aux[::-1,:],(int(nx**2),)) # Vertical flip
        MI2[:,cont + 3]=np.reshape(util.invert(aux),(int(nx**2),)) # Color Inversion 
        
    return MI2