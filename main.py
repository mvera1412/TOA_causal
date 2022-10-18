import os
import torch
from tqdm import tqdm
import numpy as np
from TOA.mbfdunetln import MBPFDUNet
from torch.optim.lr_scheduler import MultiStepLR
from TOA.train import createForwMat
from utils.causal_utils import train, validation, testing, computing_metrics, load_traindataset, load_testdataset, \
    load_ckp
from utils.noncausal_utils import load_traindataset_nc, train_nc
from train_algorithms.IRMv1.algorithm import build_lambda_map
import wandb

val_percent = 440.0 / 2216.0  # 440 para validacion, 1776 para train
le = 5  # cantidad de environments
epochs = 50
# cache_dir = '../data/' # Donde estan los datos y donde se van a guardar los modelosç
cache_dir = "datos_toa"
fecha = '120822_15'
alphas = [1e-4, 5e-4, 1e-3]
bs = [1, 2, 3]  # per environment
taus = [0.4, 0.8]

os.environ["WANDB_API_KEY"] = os.getenv("MY-API-KEY")


def setup_wandb():
    config = {
        "lr": 0.1,
        "batch_size": 5,
        "epochs": epochs,
    }
    project = "my-test-project"
    notes = "test using main"
    tags = ["tag1"]
    wandb.init(config=config, project=project, tags=tags, notes=notes)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device to be used: {device}")

    # setup_wandb()

    ##Loss
    loss_fn = torch.nn.MSELoss()

    ##TOA matrix
    Ao = createForwMat()
    Ao = torch.as_tensor(Ao).type(torch.float32)
    Ao = Ao.to(device=device)

    ##Files
    ckp_last = cache_dir + 'mbfdunetln' + fecha + '.pth'  # name of the file of the saved weights of the trained net
    ckp_best = cache_dir + 'mbfdunetln_best' + fecha + '.pth'
    checkpoint = {'valid_loss_min': np.inf}

    # Entrenamiento red causal
    lambda_map: dict = build_lambda_map()
    epoch0 = 0
    algorithm = "IRMv1"
    param_dict: dict
    for batchsize in bs:
        train_loaders, val_loader = load_traindataset(cache_dir, val_percent, batchsize, val_batchsize=40, le=le)
        for lr in alphas:
            for algorithm_param in taus:
                model = MBPFDUNet().to(device=device)
                checkpoint['state_dict'] = model.state_dict()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                checkpoint['learning_rate'] = lr
                checkpoint['batchsize'] = batchsize
                checkpoint['agreement_threshold'] = algorithm_param
                checkpoint['optimizer'] = optimizer.state_dict()
                checkpoint['epoch'] = epoch0
                lr_scheduler = MultiStepLR(optimizer, milestones=[le * epochs * 3 // 4], gamma=0.1)
                lr_scheduler = None
                if algorithm == "ANDMask":
                    param_dict = {'agreement_threshold': algorithm_param,
                                  'method': 'and_mask',
                                  'scale_grad_inverse_sparsity': 1}
                elif algorithm == "IRMv1":
                    param_dict = {'irm_lambda': algorithm_param,
                                  'penalty_anneal_epochs': 30,
                                  'l2_reg': 0.001}
                    param_dict = {'use_epoch_lambda_map': True,
                                  'epoch_lambda_map': lambda_map,
                                  'l2_reg': 0.001}
                for epoch in tqdm(range(epoch0 + 1, epochs + 1)):
                    train(model, device, train_loaders, optimizer, n_agreement_envs=le, Ao=Ao, loss_fn=loss_fn,
                          scheduler=lr_scheduler, epoch=epoch, algorithm=algorithm, **param_dict)
                    valid_loss_min, valid_loss, env_losses_dict = validation(
                        model, device, val_loader, optimizer, loss_fn, Ao, checkpoint, ckp_last, ckp_best, fecha,
                        metrics=True
                    )
                    val_metrics = {'validation_loss': valid_loss}
                    for env, loss in env_losses_dict.items():
                        val_metrics[f"env_{env}"] = loss
                    wandb.log(val_metrics)
                    checkpoint['epoch'] = epoch
                    checkpoint['valid_loss_min'] = valid_loss_min

    model, optimizer, best_epoch, valid_loss_min, best_lr, best_bs, best_threshold = load_ckp(ckp_best, model,
                                                                                              optimizer)

    # Entrenamiento red benchmark
    ckp_benchmark_last = cache_dir + 'benchmark' + fecha + '.pth'
    ckp_benchmark_best = cache_dir + 'benchmark_best' + fecha + '.pth'
    checkpoint_nc = {'valid_loss_min': np.inf, 'agreement_threshold': 0.0}
    epoch0 = 0
    for batchsize in bs:
        train_loader_nc, val_loader_nc = load_traindataset_nc(cache_dir, val_percent, batchsize * le, val_batchsize=40,
                                                              le=le)
        for lr in alphas:
            model_nc = MBPFDUNet().to(device=device)
            checkpoint_nc['state_dict'] = model_nc.state_dict()
            optimizer_nc = torch.optim.Adam(model_nc.parameters(), lr=lr)
            checkpoint_nc['learning_rate'] = lr
            checkpoint_nc['batchsize'] = batchsize
            checkpoint_nc['optimizer'] = optimizer_nc.state_dict()
            checkpoint_nc['epoch'] = epoch0
            lr_scheduler_nc = MultiStepLR(optimizer_nc, milestones=[le * epochs * 3 // 4], gamma=0.1)
            for epoch in tqdm(range(epoch0 + 1, epochs + 1)):
                train_nc(model_nc, device, train_loader_nc, optimizer_nc, Ao=Ao, loss_fn=loss_fn,
                         scheduler=lr_scheduler_nc)
                checkpoint_nc['epoch'] = epoch
                checkpoint_nc['valid_loss_min'] = validation(model_nc, device, val_loader_nc, optimizer_nc, loss_fn, Ao,
                                                             checkpoint_nc, ckp_benchmark_last, ckp_benchmark_best,
                                                             fecha)

    model_nc, optimizer_nc, best_epoch_nc, vlm_nc, lr_nc, bs_nc, at_nc = load_ckp(ckp_benchmark_best, model_nc,
                                                                                  optimizer_nc)

    # Testing
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
            a, b, c, d = computing_metrics(data_test[0].to("cpu"), data_test[1].to("cpu"), Ao.to(device="cpu"), model,
                                           model_nc)
            SSIM[j].append(a)
            PC[j].append(b)
            RMSE[j].append(c)
            PSNR[j].append(d)

    testing(np.array(SSIM), np.array(PC), np.array(RMSE), np.array(PSNR), test_loaders, Ao.to(device="cpu"), model,
            model_nc)
