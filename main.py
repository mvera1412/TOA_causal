import torch
import pickle
import gdown
import os

if __name__ == '__main__':

	batchsize = 10
	cache_dir = 'data/cache/'
	if torch.cuda.is_available():
	    device = torch.device("cuda")
	else:
	    device = torch.device("cpu")
	print(f"Device to be used: {device}")
	url = 'https://drive.google.com/file/d/1mAhSP4sqcmtJlwKUDtFxcb4Aqi_W0Shs'
	output = 'train_TOA'
	gdown.download(url, output, quiet=False)
	with open("train_TOA", "rb") as fp:   # Unpickling
		train_loaders = pickle.load(fp)
	os.remove("./train_TOA") 
	train_loaders
