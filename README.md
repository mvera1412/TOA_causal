# TOA_causal

batchsize = 3 #per environment
epochs = 50
lr = 1e-4
wd= 1e-6
agreement_threshold = 0.3 
fecha = '260722_15'
E0 - ANDMask:
SSIM 0.472 - PC 0.774 - RMSE 0.151 - PSNR 16.756
E0 - Benchmark:
SSIM 0.517 - PC 0.779 - RMSE 0.144 - PSNR 17.159
E1 - ANDMask:
SSIM 0.809 - PC 0.957 - RMSE 0.070 - PSNR 23.720
E1 - Benchmark:
SSIM 0.841 - PC 0.963 - RMSE 0.061 - PSNR 24.916

############################################################### 

batchsize = 2 #per environment
epochs = 50
lr = 1e-3
wd= 1e-6
agreement_threshold = 0.5 
fecha = '280722_10'
E0 - ANDMask:
SSIM:  0.493  PC:  0.755  RMSE:  0.142  PSNR:  17.225
E0 - Benchmark:
SSIM:  0.397  PC:  0.705  RMSE:  0.166  PSNR:  15.901
E1 - ANDMask:
SSIM:  0.822  PC:  0.96  RMSE:  0.064  PSNR:  24.497
E1 - Benchmark:
SSIM:  0.801  PC:  0.952  RMSE:  0.071  PSNR:  23.636

############################################################### 

batchsize = 2,3 #per environment
epochs = 50
lr = 1e-2,1e-3,1e-4
wd= 0.0,1e-6
agreement_threshold = 0.4,0.8 
fecha = '280722_16'

E0 - ANDMask:
SSIM:  0.592  PC:  0.834  RMSE:  0.117  PSNR:  18.929
E0 - Benchmark:
SSIM:  0.574  PC:  0.832  RMSE:  0.121  PSNR:  18.689
E1 - ANDMask:
SSIM:  0.878  PC:  0.976  RMSE:  0.047  PSNR:  27.181
E1 - Benchmark: 
SSIM:  0.838  PC:  0.963  RMSE:  0.06  PSNR:  25.113
