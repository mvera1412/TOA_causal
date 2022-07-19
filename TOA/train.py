# Importing the necessary libraries:

#import os
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
from TOA.model_based_matrix import build_matrix, SensorMaskCartCircleArc

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.optim import Adam

from TOA.mbfdunetln import MBPFDUNet


# ---------------------------------------------------------------------------
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

# ---------------------------------------------------------------------------
def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    loss = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], loss

    
# ---------------------------------------------------------------------------
def gettraindata(cache_dir):
    
    print('Obtaining data for training...')
    
    #cache_dir = '../data/cache'

    X0 = np.load(os.path.join(cache_dir, 'X0.npy')) # Noisy sinogram
    X1 = np.load(os.path.join(cache_dir, 'X1.npy')) 
    X2 = np.load(os.path.join(cache_dir, 'X2.npy'))
    X3 = np.load(os.path.join(cache_dir, 'X3.npy'))
    X4 = np.load(os.path.join(cache_dir, 'X4.npy'))
    X = np.append(X0,X1,axis=0); X = np.append(X,X2,axis=0); X = np.append(X,X3,axis=0); X = np.append(X,X4,axis=0)
    del X0,X1,X2,X3,X4
    Y0 = np.load(os.path.join(cache_dir, 'Y0.npy')) # True image
    Y1 = np.load(os.path.join(cache_dir, 'Y1.npy')) 
    Y2 = np.load(os.path.join(cache_dir, 'Y2.npy'))
    Y3 = np.load(os.path.join(cache_dir, 'Y3.npy'))
    Y4 = np.load(os.path.join(cache_dir, 'Y4.npy'))
    Y = np.append(Y0,Y1,axis=0); Y = np.append(Y,Y2,axis=0); Y = np.append(Y,Y3,axis=0); Y = np.append(Y,Y4,axis=0)
    del Y0,Y1,Y2,Y3,Y4
    #Z0 = np.load(os.path.join(cache_dir, 'Z0.npy')) # True sinogram
    #Z1 = np.load(os.path.join(cache_dir, 'Z1.npy'))
    #Z2 = np.load(os.path.join(cache_dir, 'Z2.npy'))
    #Z3 = np.load(os.path.join(cache_dir, 'Z3.npy'))
    #Z4 = np.load(os.path.join(cache_dir, 'Z4.npy'))
    #Z = np.append(Z0,Z1,axis=0); Z = np.append(Z,Z2,axis=0); Z = np.append(Z,Z3,axis=0); Z = np.append(Z,Z4,axis=0);
    #del Z0,Z1,Z2,Z3,Z4
    
    # Shuffle data
    indpat = np.arange(0, X.shape[0], dtype=int)  # indice patrón
    ida = np.random.permutation(indpat)
    X = X[ida,:,:]
    Y = Y[ida,:,:]
    #Z = Z[ida,:,:]
        
    X=X.astype(np.float32)
    Y=Y.astype(np.float32)
    #Z=Z.astype(np.float32)
    print('done')
    
    return X,Y #,Z

# ---------------------------------------------------------------------------
class OAImageDataset(Dataset):
    def __init__(self, X, Y):
        super(OAImageDataset, self).__init__()
        self.X = X
        self.Y = Y

    def __getitem__(self, item):
        return self.X[item, :, :], self.Y[item, :, :]

    def __len__(self):
        return self.X.shape[0]

# ---------------------------------------------------------------------------
def get_trainloader(X, Y, val_percent, batch_size): 
        
    dataset_train = OAImageDataset(X, Y)
    
    # Split into train / validation partitions
    n_val = int(len(dataset_train) * val_percent)
    n_train = len(dataset_train) - n_val
    train_set, val_set = random_split(dataset_train, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    # Create data loaders
    #loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True) # for local uncomment this
    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True) # for google_colab uncomment this
    train_loader = DataLoader(train_set, shuffle=True,  drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args) #drop_last=True, drop the last batch if the dataset size is not divisible by the batch size.
    
    return train_loader, val_loader, n_train, n_val

# ---------------------------------------------------------------------------
def createForwMat(): # 
    print('Creating Forward Model-based Matrix without position uncertainty')
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
    Ao = build_matrix(nx,dx,Ns,posSens,1,1,vs,t,True,True,True,True,tlp=2*dx/vs)
    print('done')
    Ao=Ao.astype(np.float32)
    print('done')
    
    return Ao

# ---------------------------------------------------------------------------
def applyInvMat(x, Ao, dimS, dimI): # [Ao] = (16384,4096)
    x = torch.squeeze(x,1) # (-1,32,512)
    x = torch.reshape(x,(dimS[0],int(dimS[2]*dimS[3]))) # (-1,16384)
    y = torch.matmul(Ao.T,x.T).T # ((4096,16384) @ (16384,-1)).T = (-1,4096)
    y = torch.reshape(y,(dimI[0],dimI[2],dimI[3])) # (-1,64,64)
    y = torch.unsqueeze(y,1) # (-1,1,64,64)
    
    return y

# ---------------------------------------------------------------------------
def applyForwMat(y, Ao, dimS, dimI):
    y = torch.squeeze(y,1) # (-1,64,64)
    y = torch.reshape(y,(dimI[0],int(dimI[2]*dimI[3]))) # (-1,4096)
    x = torch.matmul(Ao,y.T).T # ((16384,4096) @ (4096,-1)).T = (-1,16384)
    x = torch.reshape(x,(dimS[0],dimS[2],dimS[3])) # (-1,32,512)
    x = torch.unsqueeze(x,1) # (-1,1,32,512)
        
    return x

# ---------------------------------------------------------------------------
def train_mbfdunetln(batch_size,epochs,continuetrain,plotresults,WandB,fecha):
    
    cache_dir = '../data/cache/'
    
    # Net Main Parameters
    lr = 1e-4
    beta1 = 0.9 #0.5
    #traindata = False
    ckp_last='mbfdunetln' + fecha + '.pth' # name of the file of the saved weights of the trained net
    ckp_best='mbfdunetln_best' + fecha + '.pth'
    
    # 1. Set device
    ngpu = 1 # number og GPUs available. Use 0 for CPU mode.
    device = ""
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device to be used: {device}")
    
    # 1. Create de network
    net = MBPFDUNet().to(device=device)
    
    # Number of net parameters
    NoP = sum(p.numel() for p in net.parameters())
    print(f"The number of parameters of the network to be trained is: {NoP}")    
    
    # 2. Define loss function and optimizer and the the learning rate scheduler
    optimizer = Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=2,threshold=0.005,eps=1e-6,verbose=True)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5,verbose=True)
    LossFn = nn.MSELoss()
    #LossFn = CharbonnierLoss()
    
    # Handle multi-gpu if desired
    if (device.type == "cuda") and (ngpu > 1):
        net = nn.DataParallel(net, list(range(ngpu)))
    
    # 3. Create datase
    # Create Model-based Matrix
    Ao = createForwMat()
    Ao = torch.as_tensor(Ao).type(torch.float32)
    Ao = Ao.to(device=device)
    
    # Get data
    X,Y = gettraindata(cache_dir)
    
    # 4. Create data loader
    val_percent = 0.2
    X = torch.as_tensor(X).type(torch.float32) 
    Y = torch.as_tensor(Y).type(torch.float32)
    train_loader, val_loader, n_train, n_val = get_trainloader(X, Y, val_percent, batch_size)
    
    # 5. Initialize logging and initialize weights or continue a previous training 
    logfilename='TrainingLog_MBFDUNetLN.log'
    
    if continuetrain:
        net, optimizer, last_epoch, valid_loss_min = load_ckp(ckp_last, net, optimizer)
        print('Values loaded:')
        #print("model = ", net)
        print("optimizer = ", optimizer)
        print("last_epoch = ", last_epoch)
        print("valid_loss_min = ", valid_loss_min)
        print("valid_loss_min = {:.6f}".format(valid_loss_min))
        start_epoch = last_epoch + 1
        lr = optimizer.param_groups[0]['lr']
        logging.basicConfig(filename=logfilename,format='%(asctime)s - %(message)s', level=logging.INFO)
        logging.info(f'''Continuing training:
            Epochs:                {epochs}
            Batch size:            {batch_size}
            Initial learning rate: {lr}
            Training size:         {n_train}
            Validation size:       {n_val}
            Device:                {device.type}
            ''')
    else:
        # Apply the weights_init function to randomly initialize all weights
        net.apply(initialize_weights)
        start_epoch = 1
        valid_loss_min = 100
        logging.basicConfig(filename=logfilename, filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO)
        logging.info(f'''Starting training:
            Epochs:                {epochs}
            Batch size:            {batch_size}
            Initial learning rate: {lr}
            Training size:         {n_train}
            Validation size:       {n_val}
            Device:                {device.type}
            ''')
    
    if WandB:
        experiment =wandb.init(project='mbfdunetln_oa', entity='dl_oa_fiuba')
        experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=lr, val_percent=val_percent))
    # Print model
    # print(net)

    # 6. Begin training
    TLV=np.zeros((epochs,)) #vector to record the train loss per epoch 
    VLV=np.zeros((epochs,)) #vector to record the validation loss per epoch
    EV=np.zeros((epochs,)) # epoch vector to plot later
    global_step = 0
    
    #for epoch in range(epochs):
    for epoch in range(start_epoch, start_epoch+epochs):
        net.train() # Let pytorch know that we are in train-mode
        epoch_loss = 0.0
        epoch_val_loss = 0.0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs+start_epoch-1}', unit='sino') as pbar:
            for x,y in train_loader:
                # clear the gradients
                optimizer.zero_grad(set_to_none=True)
                # input and truth to device
                x = x.to(device=device)
                x = torch.unsqueeze(x,1)
                x = x.type(torch.float32)
                #
                dimS = x.shape # (-1,1,128,512)
                dimI = (dimS[0],dimS[1],64,64) # (-1,1,64,64)
                f0 = applyInvMat(x,Ao,dimS,dimI) # (-1,1,64,64)
                g1 = applyForwMat(f0,Ao,dimS,dimI) # (-1,1,32,512)
                Dg = g1 - x # (-1,1,128,512)
                Df = applyInvMat(Dg,Ao,dimS,dimI) # (-1,1,64,64)
                #
                y = y.to(device=device) # (-1,1,64,64)
                # compute the model output
                pred = net(f0,Df)
                #pred = net(x.type(torch.float32))
                pred = torch.squeeze(pred,1)
                # calculate loss
                loss = LossFn(pred, y.type(torch.float32))
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()
                
                pbar.update(x.shape[0])
                global_step += 1
                
                epoch_loss += loss.item()
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})
            
            epoch_train_loss = epoch_loss / len(train_loader)
        # Scheduler Step
        #scheduler.step()
        # Evaluation round
        with torch.no_grad():
            for xv, yv in tqdm(val_loader, total=len(val_loader), desc='Validation round', position=0, leave=True):
                # input and truth to device
                xv = xv.to(device=device)
                xv = torch.unsqueeze(xv,1)
                xv = xv.type(torch.float32)
                #
                dimS = xv.shape # (-1,1,32,512)
                dimI = (dimS[0],dimS[1],64,64) # (-1,1,64,64)
                f0v = applyInvMat(xv,Ao,dimS,dimI) # (-1,1,64,64)
                g1v = applyForwMat(f0v,Ao,dimS,dimI) # (-1,1,32,512)
                Dgv = g1v - xv # (-1,1,128,512)
                Dfv = applyInvMat(Dgv,Ao,dimS,dimI) # (-1,1,64,64)
                #
                yv = yv.to(device=device) # (-1,1,64,64)
                # compute the model output
                #pred = net(xv.unsqueeze(1).type(torch.float32))
                predv = net(f0v,Dfv)
                predv = predv.squeeze(1)
                # calculate loss
                loss = LossFn(predv, yv.type(torch.float32))
                epoch_val_loss += loss.item()
        
        epoch_val_loss = epoch_val_loss / len(val_loader)
        # Scheduler ReduceLROnPlateau
        scheduler.step(epoch_val_loss)
        # Scheduler StepLR
        #scheduler.step()
        
        # logging validation score per epoch
        logging.info(f'''Epoch: {epoch} - Validation score: {np.round(epoch_val_loss,5)}''')
        
        # print training/validation statistics 
        #print('\n Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        #    epoch,
        print('\n Training Loss: {:.5f} \tValidation Loss: {:.5f}'.format(
            epoch_train_loss,
            epoch_val_loss
            ))
        
        # Loss vectors for plotting results
        TLV[epoch-start_epoch]=epoch_train_loss
        VLV[epoch-start_epoch]=epoch_val_loss
        EV[epoch-start_epoch]=epoch
        
        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch,
            'valid_loss_min': epoch_val_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        # save checkpoint
        save_ckp(checkpoint, False, ckp_last, ckp_best)
        
        
        # save the model if validation loss has decreased
        if epoch_val_loss <= valid_loss_min:
            print('\n Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss),'\n')
            # save checkpoint as best model
            save_ckp(checkpoint, True, ckp_last, ckp_best)
            valid_loss_min = epoch_val_loss
            logging.info(f'Val loss deccreased on epoch {epoch}!')
        
        # Logging wandb
        if WandB:
            experiment.log({
                'train loss': epoch_train_loss,
                'val loss': epoch_val_loss,
                'epoch': epoch
            })
    
    if WandB:
        wandb.finish()
    
    del x,y,xv,yv,pred,predv,Ao,f0,g1,Dg,Df,f0v,g1v,Dgv,Dfv
    
    if plotresults:
        plt.figure();
        plt.grid(True,linestyle='--')
        plt.xlabel('epoch'); plt.ylabel('Loss')
        plt.plot(EV,TLV,'--',label='Train Loss')
        plt.plot(EV,VLV,'-',label='Val Loss')
        plt.legend(loc='best',shadow=True, fontsize='x-large')
    
    return EV,TLV,VLV

# ---------------------------------------------------------------------------
def predict_out(net, inp, Ao, device):
    x = torch.as_tensor(inp)
    x = x.to(device=device)
    x = torch.unsqueeze(x,1)
    x = x.type(torch.float32)
    Ao = torch.as_tensor(Ao).type(torch.float32)
    Ao = Ao.to(device=device)
    #
    dimS = x.shape # (-1,1,32,512)
    dimI = (dimS[0],dimS[1],64,64) # (-1,1,64,64)
    f0 = applyInvMat(x,Ao,dimS,dimI) # (-1,1,64,64)
    g1 = applyForwMat(f0,Ao,dimS,dimI) # (-1,1,32,512)
    Dg = g1 - x # (-1,1,32,512)
    Df = applyInvMat(Dg,Ao,dimS,dimI) # (-1,1,64,64)
    with torch.no_grad():
        pred = net(f0,Df) 
        pred = pred.squeeze(1) # (-1,64,64)
    
    return pred.detach().to("cpu").numpy()
                
# --------------------------------------------------------------------------- 
def initialize_weights(m):
    if isinstance(m,(nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data,0.0,0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data,0)
    elif isinstance(m,(nn.BatchNorm2d,nn.LayerNorm)):
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)

# ---------------------------------------------------------------------------
#if __name__=='__main__':
#    test()
