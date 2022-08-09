import torch
import os
from tqdm import tqdm

from TOA.mbfdunetln import MBPFDUNet
from ANDMask.adam_flexible_weight_decay import AdamFlexibleWeightDecay
from TOA.train import createForwMat
from utils.causal_utils import train,validation,testing,load_traindataset,load_testdataset,load_ckp
from utils.noncausal_utils import load_traindataset_nc,train_nc

batchsize = 3 #per environment
val_percent = 440.0/2216.0 # 440 para validacion, 1776 para train
le = 2
epochs = 50
lr = 1e-4
agreement_threshold = 0.3 
cache_dir = 'data/cache/'
fecha = ''
continue_training = False
    
if __name__ == '__main__':
##DEVICE
	if torch.cuda.is_available():
	    device = torch.device("cuda")
	else:
	    device = torch.device("cpu")
	print(f"Device to be used: {device}")

##Load train dataset	
	train_loaders, val_loader = load_traindataset(cache_dir,val_percent,batchsize,val_batchsize=40,le = le)

##Model and optimizer
	model = MBPFDUNet().to(device=device)
	optimizer = AdamFlexibleWeightDecay(model.parameters(),lr=lr,weight_decay_order='before',weight_decay=wd)
	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[le * epochs * 3 // 4],gamma=0.1)
	loss_fn = torch.nn.MSELoss()

##TOA matrix
	Ao = createForwMat()
	Ao = torch.as_tensor(Ao).type(torch.float32)
	Ao = Ao.to(device=device)

##Training
	ckp_last = cache_dir + 'mbfdunetln' + fecha + '.pth' # COMPARAR CON LA VERSION DE TUPAC
	ckp_best = cache_dir + 'mbfdunetln_best' + fecha + '.pth'
	if continue_training:
		model, optimizer, epoch0, valid_loss_min = load_ckp(ckp_last, model, optimizer)
	else:
		epoch0 = 0
		valid_loss_min = np.inf

	for epoch in tqdm(range(epoch0 + 1, epochs + 1)):
		train(model,device,train_loaders,optimizer,n_agreement_envs=le,Ao=Ao,loss_fn=loss_fn,agreement_threshold=agreement_threshold,scheduler=lr_scheduler)
		valid_loss_min = validation(model,device, val_loader, optimizer, loss_fn, Ao, valid_loss_min, epoch, ckp_last, ckp_best)

	model, optimizer, best_epoch, valid_loss_min = load_ckp(ckp_best, model, optimizer)

##Training benchmark model
	ckp_benchmark_last = cache_dir + 'benchmark.pth'
	ckp_benchmark_best = cache_dir + 'benchmark_best.pth'
	model_nc = MBPFDUNet().to(device=device)
	optimizer_nc = torch.optim.Adam(model_nc.parameters(),lr=lr,weight_decay=wd)
	if continue_training:
		model_nc, optimizer_nc, epoch0, vlm_nc = load_ckp(ckp_benchmark_last, model_nc, optimizer_nc)
	else:
		epoch0 = 0
		vlm_nc = np.inf
	train_loader_nc, val_loader_nc = load_traindataset_nc(cache_dir,val_percent,batchsize*le,val_batchsize=40,le = le)
	for epoch in tqdm(range(epoch0 + 1, epochs + 1)):
		train_nc(model_nc,device,train_loader_nc,optimizer_nc,Ao=Ao,loss_fn=loss_fn,scheduler=lr_scheduler)
		vlm_nc = validation(model_nc, device, val_loader_nc, optimizer_nc, loss_fn, Ao, vlm_nc, epoch, ckp_benchmark_last, ckp_benchmark_best)

	model_nc, optimizer_nc, best_epoch_nc, vlm_nc = load_ckp(ckp_benchmark_best, model_nc, optimizer_nc)

##Testing
	test_loaders = load_testdataset(cache_dir)
	le_test = len(test_loaders)
	SSIM = [[] for _ in range(le_test)] 
	PC = [[] for _ in range(le_test)] 
	RMSE = [[] for _ in range(le_test)]
	PSNR = [[] for _ in range(le_test)]
	for j in range(le_test):
		iterator = iter(test_loaders[j])
		while 1:
			try:
				data_test = next(iterator)
			except StopIteration:
				break  
			a,b,c,d=computing_metrics(data_test[0].to("cpu"),data_test[1].to("cpu"),Ao.to(device="cpu"),model,model_nc)
			SSIM[j].append(a)
			PC[j].append(b)
			RMSE[j].append(c)
			PSNR[j].append(d) 

	testing(np.array(SSIM),np.array(PC),np.array(RMSE),np.array(PSNR),test_loaders, Ao.to(device="cpu"),model, model_nc) 
