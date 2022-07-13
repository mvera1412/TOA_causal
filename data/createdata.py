from data.augmentation_letter import augmentate_letterdata
from data.augmentation_retina import augmentate_retina
from data.image_loader import imretina
from data.image_loader import imletnum
from data.model_based_matrix import build_matrix, SensorMaskCartCircleArc
import numpy as np
import os

# ---------------------------------------------------------------------------
def create_trainatestdata():

    print('Obtaining data for training...')
    
    cache_dir = '../data/cache/'
    
    IM1 = imretina('../data/retina/')
    IM2 = imletnum('../data/letnum/')
    IM3 = augmentate_letterdata(IM2)
    IM4 = augmentate_retina(IM1)
    IM = np.append(IM4,IM3,axis=1)
    
    # Freeing memory space
    del IM1,IM2,IM3,IM4
    
    # Shuffle images
    indpat = np.arange(0, IM.shape[1], dtype=int)  # indice patrón
    ida = np.random.permutation(indpat)
    IM = IM[:, ida]
    
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
    
    # sensor position error [%] 
    spe=np.array([3.8851, -0.5203, -0.18240, 0.06080, 0.00034]) # %
    spetest = np.array([2.9516, 0.00012]) # %
    
    # set aside images for testing
    aux = IM
    IMtest = aux[:,0:int(aux.shape[1]*0.1)]
    IM = aux[:,int(aux.shape[1]*0.1):-1]
    del aux
    
    print('Creating Forward Model-based Matrix without position uncertainty')
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    Ao = build_matrix(nx,dx,Ns,posSens,1,1,vs,t,True,True,True,True,tlp=2*dx/vs)
    print('done')
    
    # DATASET FOR TRAINING 
    
    # Divide IM into 5 batches
    nb = int(IM.shape[1]/len(spe))
    
    # Number of different SNR 
    nsnr = 4
    
    # Data matrix initialization
    X = np.zeros((nsnr*nb, Ns, Nt)) # corrupted sinogram
    Y = np.zeros((nsnr*nb, nx, nx)) # true image
    Z = np.zeros((nsnr*nb, Ns, Nt)) # true sinogram

    SNR = np.zeros(nsnr*nb) # Signal to noise ratio of the corrupted sinogram
    SNRa1 = np.zeros(Ns) # SNR of each detector
    SNRa2 = np.zeros(Ns)
    SNRa3 = np.zeros(Ns)
    SNRa4 = np.zeros(Ns)
    
    # Iteration for each image or sinogram for training
    for i1 in range(0,len(spe)):
        cont = -nsnr # Counter index
        IMb = IM[:,i1:(i1+1)*nb]
        print('Creating Forward Model-based Matrix with position uncertainty')
        dsae = dsa + dsa*spe[i1]/100 
        posSens = SensorMaskCartCircleArc(dsae,arco,Ns) # position of the center of the detectors (3,Ns) [m]
        Ae = build_matrix(nx,dx,Ns,posSens,1,1,vs,t,True,True,True,True,tlp=2*dx/vs)
        #print('done')
        for i2 in range(0, nb):
            cont = cont + nsnr
            h = IMb[0:,i2]
            h=h.astype(np.float32)
            yo = Ao @ h
            ye = Ae @ h
            myo = np.reshape(yo,(Ns,Nt))
            mye = np.reshape(ye,(Ns,Nt))
            
            rm = 0  # white noise mean value
            nru = np.random.uniform(0, 0.0001, 1)[0]
            nru2 = np.random.uniform(0, 0.001, 1)[0]
            nru3 = np.random.uniform(0, 0.01, 1)[0]
            nru4 = np.random.uniform(0, 0.1, 1)[0]
            rstd = nru * np.max(np.abs(yo))  # noise standard deviation 
            rstd2 = nru2 * np.max(np.abs(yo))
            rstd3 = nru3 * np.max(np.abs(yo))
            rstd4 = nru4 * np.max(np.abs(yo))
            
            ruido = np.random.normal(rm, rstd, (Ns,Nt))
            ruido2 = np.random.normal(rm, rstd2, (Ns, Nt))
            ruido3 = np.random.normal(rm, rstd3, (Ns, Nt))
            ruido4 = np.random.normal(rm, rstd4, (Ns, Nt))
            
            for i3 in range(Ns):
                SNRa1[i3]= 20 * np.log10(np.max(np.abs(mye[i3,:])) / np.abs(np.std(ruido[i3,:])))  # high SNR
                SNRa2[i3]= 20 * np.log10(np.max(np.abs(mye[i3,:])) / np.abs(np.std(ruido2[i3,:]))) 
                SNRa3[i3]= 20 * np.log10(np.max(np.abs(mye[i3,:])) / np.abs(np.std(ruido3[i3,:]))) 
                SNRa4[i3]= 20 * np.log10(np.max(np.abs(mye[i3,:])) / np.abs(np.std(ruido4[i3,:])))  # low SNR
            
            SNR[cont] = np.mean(SNRa1) # Sinogram SNR with noise 1
            SNR[cont + 1] = np.mean(SNRa2) # Sinogram SNR with noise 2
            SNR[cont + 2] = np.mean(SNRa3) # Sinogram SNR with noise 3
            SNR[cont + 3] = np.mean(SNRa4) # Sinogram SNR with noise 4
            
            print('Batch: ', i1+1, '  Sinogram: ', cont + nsnr-1, ' de ', nsnr*nb, '  Image: ', i2 + 1, ' SNR(dB): ', int(SNR[cont + nsnr-1]))
             
            X[cont, 0:, 0:] = mye + ruido
            Y[cont, 0:, 0:] = np.reshape(h, (nx, nx))
            Z[cont, 0:, 0:] = myo
            
            X[cont + 1, 0:, 0:] = mye + ruido2
            Y[cont + 1, 0:, 0:] = np.reshape(h, (nx, nx))
            Z[cont + 1, 0:, 0:] = myo
            
            X[cont + 2, 0:, 0:] = mye + ruido3
            Y[cont + 2, 0:, 0:] = np.reshape(h, (nx, nx))
            Z[cont + 2, 0:, 0:] = myo
            
            X[cont + 3, 0:, 0:] = mye + ruido4
            Y[cont + 3, 0:, 0:] = np.reshape(h, (nx, nx))
            Z[cont + 3, 0:, 0:] = myo
        
        # Shuffle data
        indpat = np.arange(0, X.shape[0], dtype=int)  
        ida = np.random.permutation(indpat)
        X = X[ida, :, :]
        Y = Y[ida, :, :]
        Z = Z[ida, :, :]
        SNR = SNR[ida]
        
        print('Saving training data...')
        np.save(os.path.join(cache_dir, 'X'+str(i1)), X)
        np.save(os.path.join(cache_dir, 'Y'+str(i1)), Y)
        np.save(os.path.join(cache_dir, 'Z'+str(i1)), Z)
        np.save(os.path.join(cache_dir, 'SNR'+str(i1)), SNR)

    # DATASET FOR TESTING
    
    # Divide IM into 5 batches
    nb = int(IMtest.shape[1]/len(spetest))
    
    # Number of different SNR 
    nsnr = 4
    
    # Data matrix initialization
    X = np.zeros((nsnr*nb, Ns, Nt)) # corrupted sinogram
    Y = np.zeros((nsnr*nb, nx, nx)) # true image
    Z = np.zeros((nsnr*nb, Ns, Nt)) # true sinogram

    SNR = np.zeros(nsnr*nb) # Signal to noise ratio of the corrupted sinogram
    SNRa1 = np.zeros(Ns) # SNR of each detector
    SNRa2 = np.zeros(Ns)
    SNRa3 = np.zeros(Ns)
    SNRa4 = np.zeros(Ns)
    
    # Iteration for each image or sinogram for testing
    for i1 in range(0,len(spetest)):
        cont = -nsnr # Counter index
        IMb = IMtest[:,i1:(i1+1)*nb]
        print('Creating Forward Model-based Matrix with position uncertainty')
        dsae = dsa + dsa*spetest[i1]/100 
        posSens = SensorMaskCartCircleArc(dsae,arco,Ns) # position of the center of the detectors (3,Ns) [m]
        Ae = build_matrix(nx,dx,Ns,posSens,1,1,vs,t,True,True,True,True,tlp=2*dx/vs)
        #print('done')
        for i2 in range(0, nb):
            cont = cont + nsnr
            h = IMb[0:,i2]
            h=h.astype(np.float32)
            yo = Ao @ h
            ye = Ae @ h
            myo = np.reshape(yo,(Ns,Nt))
            mye = np.reshape(ye,(Ns,Nt))
            
            rm = 0  # white noise mean value
            nru = np.random.uniform(0, 0.0001, 1)[0]
            nru2 = np.random.uniform(0, 0.001, 1)[0]
            nru3 = np.random.uniform(0, 0.01, 1)[0]
            nru4 = np.random.uniform(0, 0.1, 1)[0]
            rstd = nru * np.max(np.abs(yo))  # noise standard deviation 
            rstd2 = nru2 * np.max(np.abs(yo))
            rstd3 = nru3 * np.max(np.abs(yo))
            rstd4 = nru4 * np.max(np.abs(yo))
            
            ruido = np.random.normal(rm, rstd, (Ns,Nt))
            ruido2 = np.random.normal(rm, rstd2, (Ns, Nt))
            ruido3 = np.random.normal(rm, rstd3, (Ns, Nt))
            ruido4 = np.random.normal(rm, rstd4, (Ns, Nt))
            
            for i3 in range(Ns):
                SNRa1[i3]= 20 * np.log10(np.max(np.abs(mye[i3,:])) / np.abs(np.std(ruido[i3,:])))  # high SNR
                SNRa2[i3]= 20 * np.log10(np.max(np.abs(mye[i3,:])) / np.abs(np.std(ruido2[i3,:]))) 
                SNRa3[i3]= 20 * np.log10(np.max(np.abs(mye[i3,:])) / np.abs(np.std(ruido3[i3,:]))) 
                SNRa4[i3]= 20 * np.log10(np.max(np.abs(mye[i3,:])) / np.abs(np.std(ruido4[i3,:])))  # low SNR
            
            SNR[cont] = np.mean(SNRa1) # Sinogram SNR with noise 1
            SNR[cont + 1] = np.mean(SNRa2) # Sinogram SNR with noise 2
            SNR[cont + 2] = np.mean(SNRa3) # Sinogram SNR with noise 3
            SNR[cont + 3] = np.mean(SNRa4) # Sinogram SNR with noise 4
            
            print('Batch: ', i1+1, '  Sinogram: ', cont + nsnr-1, ' de ', nsnr*nb, '  Image: ', i2 + 1, ' SNR(dB): ', int(SNR[cont + nsnr-1]))
             
            X[cont, 0:, 0:] = mye + ruido
            Y[cont, 0:, 0:] = np.reshape(h, (nx, nx))
            Z[cont, 0:, 0:] = myo
            
            X[cont + 1, 0:, 0:] = mye + ruido2
            Y[cont + 1, 0:, 0:] = np.reshape(h, (nx, nx))
            Z[cont + 1, 0:, 0:] = myo
            
            X[cont + 2, 0:, 0:] = mye + ruido3
            Y[cont + 2, 0:, 0:] = np.reshape(h, (nx, nx))
            Z[cont + 2, 0:, 0:] = myo
            
            X[cont + 3, 0:, 0:] = mye + ruido4
            Y[cont + 3, 0:, 0:] = np.reshape(h, (nx, nx))
            Z[cont + 3, 0:, 0:] = myo
        
        # Shuffle data
        indpat = np.arange(0, X.shape[0], dtype=int)  
        ida = np.random.permutation(indpat)
        X = X[ida, :, :]
        Y = Y[ida, :, :]
        Z = Z[ida, :, :]
        SNR = SNR[ida]
        
        print('Saving testing data...')
        np.save(os.path.join(cache_dir, 'Xtest'+str(i1)), X)
        np.save(os.path.join(cache_dir, 'Ytest'+str(i1)), Y)
        np.save(os.path.join(cache_dir, 'Ztest'+str(i1)), Z)
        np.save(os.path.join(cache_dir, 'SNRtest'+str(i1)), SNR)
       
    print('Done')