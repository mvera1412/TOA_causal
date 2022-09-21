import torch
from torch import Tensor
from typing import Callable


def mean_nll(logits, y):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)


def penalty(logits, y, loss_fn: Callable):
    """Uses dummy linear classfier w of value 1.0"""
    scale = torch.tensor(1.).requires_grad_()
    loss = loss_fn(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


def compute_grads(irm_lambda: float,
                  batch_size: int,
                  loss_fn: Callable,
                  n_envs: int,
                  model_params: list,
                  output: Tensor,
                  target: Tensor
                  ):
    """Sets gradient values using outputs, targets and loss_fn(outputs, targets).
        * optimizer.zero_grad() called before entering this function
        * optimizer.step() called right after this function returns
    """
    loss_fn = loss_fn if loss_fn else mean_nll

    outputs = output.view(n_envs, batch_size, -1)  # will iterate over each env (first dim)
    targets = target.view(n_envs, batch_size, -1)
    outputs = outputs.squeeze(-1)
    targets = targets.squeeze(-1)

    env_losses = [dict() for _ in range(n_envs)]
    for env_outputs, env_targets, losses in zip(outputs, targets, env_losses):
        losses['nll'] = loss_fn(env_outputs, env_targets)
        losses['penalty'] = penalty(env_outputs, env_targets, loss_fn)

    train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()

    pass
