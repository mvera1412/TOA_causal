# Importing the necessary libraries:
import torch
from torch import nn

import numpy as np
#import math

# ---------------------------------------------------------------------------
# Set  up a random generator see so that the experiment 
# can be replicated identically on any machine:
# torch.manual_seed(111)

# ---------------------------------------------------------------------------
class DenseLayer(nn.Module):
    def __init__(self,feature_maps,in_channels,growth_rate,data_rows,data_cols):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,feature_maps,kernel_size=1,stride=1,padding='same')
        self.conv2 = nn.Conv2d(feature_maps,growth_rate,kernel_size=3,stride=1,padding='same')
        self.relu = nn.ReLU()
        self.lnorm1 = nn.LayerNorm([feature_maps, data_rows, data_cols])
        self.lnorm2 = nn.LayerNorm([growth_rate, data_rows, data_cols])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.lnorm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.lnorm2(x)
        x = self.relu(x)
        
        return x

# ---------------------------------------------------------------------------
class DenseBlock(nn.Module):
    def __init__(self,feature_maps,growth_rate,data_rows,data_cols):
        super().__init__()
        self.dl1 = DenseLayer(feature_maps,feature_maps,growth_rate,data_rows,data_cols)
        self.dl2 = DenseLayer(feature_maps,int(feature_maps+growth_rate),growth_rate,data_rows,data_cols)
        self.dl3 = DenseLayer(feature_maps,int(feature_maps+2*growth_rate),growth_rate,data_rows,data_cols)
        self.dl4 = DenseLayer(feature_maps,int(feature_maps+3*growth_rate),growth_rate,data_rows,data_cols)
    
    def forward(self, x):
        x1 = self.dl1(x)
        x2 = torch.cat([x, x1], dim=1)
        x2 = self.dl2(x2)
        x3 = torch.cat([x, x1, x2], dim=1)
        x3 = self.dl3(x3)
        x4 = torch.cat([x, x1, x2, x3], dim=1)
        x4 = self.dl4(x4)
        x5 = torch.cat([x, x1, x2, x3, x4], dim=1)
        
        return x5
    

# ---------------------------------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels,int(in_channels/2),kernel_size=2,stride=2)
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding='same')
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        
        return x
    
# ---------------------------------------------------------------------------
class FDUNet(nn.Module):
    def __init__(self,data_rows,data_cols):
        super().__init__()
        self.conv11 = nn.Conv2d(1,16,kernel_size=3,stride=1,padding='same')
        self.conv12 = nn.Conv2d(1,16,kernel_size=3,stride=1,padding='same')
        self.db1 = DenseBlock(32,8,data_rows,data_cols)
        self.db2 = DenseBlock(64,16,int(data_rows/2),int(data_cols/2))
        self.db3 = DenseBlock(128,32,int(data_rows/4),int(data_cols/4))
        self.db4 = DenseBlock(256,64,int(data_rows/8),int(data_cols/8))
        self.db5 = DenseBlock(512,128,int(data_rows/16),int(data_cols/16))
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.up1 = UpBlock(1024,256)
        self.db6 = DenseBlock(256,64,int(data_rows/8),int(data_cols/8))
        self.up2 = UpBlock(512,128)
        self.db7 = DenseBlock(128,32,int(data_rows/4),int(data_cols/4))
        self.up3 = UpBlock(256,64)
        self.db8 = DenseBlock(64,16,int(data_rows/2),int(data_cols/2))
        self.up4 = UpBlock(128,32)
        self.db9 = DenseBlock(32,8,data_rows,data_cols)
        self.conv2 = nn.Conv2d(64,1,kernel_size=1,stride=1,padding='same')
        
    def forward(self, xi1, xi2):
        
        x11 = self.conv11(xi1) # (-1,16,64,64)
        x12 = self.conv12(xi2) # (-1,16,64,64)
        x1 = torch.cat([x11, x12], dim=1) # (-1,32,64,64)
        x1 = self.db1(x1) # (-1.64,64,64)
        x2 = self.maxpool(x1) # (-1,64,32,32)
        x2 = self.db2(x2) # (-1,128,32,32)
        x3 = self.maxpool(x2) # (-1,128,16,16)
        x3 = self.db3(x3) # (-1,256,16,16)
        x4 = self.maxpool(x3) # (-1,256,8,8)
        x4 = self.db4(x4) # (-1,512,8,8)
        x5 = self.maxpool(x4) # (-1,512,4,4)
        x5 = self.db5(x5) # (-1,1024,4,4)
        
        x6 = self.up1(x5,x4) # (-1,256,8,8)
        x6 = self.db6(x6) # (-1,512,8,8)
        x7 = self.up2(x6,x3) # (-1,128,16,16)
        x7 = self.db7(x7) # (-1,256,16,16)
        x8 = self.up3(x7,x2) # (-1,64,32,32)
        x8 = self.db8(x8) # (-1,128,32,32)
        x9 = self.up4(x8,x1) # (-1,32,64,64)
        x9 = self.db9(x9) # (-1,64,64,64)
        x10 = self.conv2(x9) # (-1,1,64,64)
        
        y = x10 + xi1
        
        return y

# ---------------------------------------------------------------------------
class MBPFDUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fdunetblock = FDUNet(64,64)
            
    def forward(self, f0, Df):
        
        f1 = self.fdunetblock(f0,Df) # (-1,1,64,64)
        y = f1 # (-1,1,64,64)

        return y