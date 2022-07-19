import torch
import ignite

from TOA.mbfdunetln import MBPFDUNet
from ANDMask.adam_flexible_weight_decay import AdamFlexibleWeightDecay
from torch.optim.lr_scheduler import MultiStepLR
from TOA.train import createForwMat
import ANDMask.and_mask_utils as and_mask_utils
from ANDMask.common import permutation_groups

def applyInvMat(x, Ao, dimS, dimI): # [Ao] = (16384,4096)
    x = torch.squeeze(x,1) # (-1,32,512)
    x = torch.reshape(x,(dimS[0],int(dimS[2]*dimS[3]))) # (-1,16384)
    y = torch.matmul(Ao.T,x.T).T # ((4096,16384) @ (16384,-1)).T = (-1,4096)
    y = torch.reshape(y,(dimI[0],dimI[2],dimI[3])) # (-1,64,64)
    y = torch.unsqueeze(y,1) # (-1,1,64,64)  
    return y

def applyForwMat(y, Ao, dimS, dimI):
    y = torch.squeeze(y,1) # (-1,64,64)
    y = torch.reshape(y,(dimI[0],int(dimI[2]*dimI[3]))) # (-1,4096)
    x = torch.matmul(Ao,y.T).T # ((16384,4096) @ (4096,-1)).T = (-1,16384)
    x = torch.reshape(x,(dimS[0],dimS[2],dimS[3])) # (-1,32,512)
    x = torch.unsqueeze(x,1) # (-1,1,32,512)    
    return x

def predicting(net, input, Ao, device):
    x = input.to(device=device)
    x = torch.unsqueeze(x,1)
    x = x.type(torch.float32)
    dimS = x.shape # (-1,1,128,512)
    dimI = (dimS[0],dimS[1],64,64) # (-1,1,64,64)
    f0 = applyInvMat(x,Ao,dimS,dimI) # (-1,1,64,64)
    g1 = applyForwMat(f0,Ao,dimS,dimI) # (-1,1,32,512)
    Dg = g1 - x # (-1,1,128,512)
    Df = applyInvMat(Dg,Ao,dimS,dimI) # (-1,1,64,64)
    pred = net.to(device=device)(f0,Df)
    return torch.squeeze(pred,1)

def train(model, device, train_loaders, optimizer, epoch,
          n_agreement_envs,
          loss_fn,
          agreement_threshold,
          scheduler):

    model.train()

    losses = []
    example_count = 0
    batch_idx = 0

    train_iterators = [iter(loader) for loader in train_loaders]
    it_groups = permutation_groups(train_iterators, n_agreement_envs)

    while 1:
        train_iterator_selection = next(it_groups)
        try:
            datas = [next(iterator) for iterator in train_iterator_selection]
        except StopIteration:
            break

        assert len(datas) == n_agreement_envs

        batch_size = datas[0][0].shape[0]
        assert all(d[0].shape[0] == batch_size for d in datas)

        inputs = [d[0].to(device) for d in datas]
        target = [d[1].to(device) for d in datas]

        inputs = torch.cat(inputs, dim=0)
        target = torch.cat(target, dim=0)

        optimizer.zero_grad()

        output = predicting(model, inputs, Ao, device)
        mean_loss, masks = and_mask_utils.get_grads(
            agreement_threshold,
            batch_size,
            loss_fn, n_agreement_envs,
            params=optimizer.param_groups[0]['params'],
            output=output,
            target=target,
            method='and_mask',
            scale_grad_inverse_sparsity=1,
        )
        optimizer.step()

        losses.append(mean_loss.item())
        example_count += output.shape[0]
        batch_idx += 1

    scheduler.step()
    
