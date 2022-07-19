import torch
import pickle
import gdown
import os
import ignite
from tqdm import tqdm

from TOA.mbfdunetln import MBPFDUNet
from ANDMask.adam_flexible_weight_decay import AdamFlexibleWeightDecay
from torch.optim.lr_scheduler import MultiStepLR
from TOA.train import createForwMat
from utils.causal_utils import train

epochs = 50
lr = 1e-4
wd= 1e-6
agreement_threshold = 0.3
   
    
if __name__ == '__main__':
##DEVICE
	if torch.cuda.is_available():
	    device = torch.device("cuda")
	else:
	    device = torch.device("cpu")
	print(f"Device to be used: {device}")

##Train dataset	
	if not(os.path.exists('train_TOA')):
		url = 'https://drive.google.com/file/d/1mAhSP4sqcmtJlwKUDtFxcb4Aqi_W0Shs'
		output = 'data/train_TOA'
		gdown.download(url, output, quiet=False)
	with open("data/train_TOA", "rb") as fp:   # Unpickling
		train_loaders = pickle.load(fp)
	#os.remove("data/train_TOA") 

##Model and optimizer
	model = MBPFDUNet().to(device=device)
	optimizer = AdamFlexibleWeightDecay(model.parameters(),lr=lr,weight_decay_order='before',weight_decay=wd)
	le = len(train_loaders)
	lr_scheduler = MultiStepLR(optimizer,milestones=[le * epochs * 3 // 4],gamma=0.1)
	loss_fn = torch.nn.MSELoss()

##TOA matrix
	Ao = createForwMat()
	Ao = torch.as_tensor(Ao).type(torch.float32)
	Ao = Ao.to(device=device)

##Training

	for epoch in tqdm(range(1, epochs + 1)):
		train(model,
              device,
              train_loaders,
              optimizer,
              epoch,
              n_agreement_envs=le,
              loss_fn=loss_fn,
              agreement_threshold=agreement_threshold,
              scheduler=lr_scheduler
              )
        
	state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
	f_path='mbfdunetln_causal.pth'
	torch.save(state, f_path)
