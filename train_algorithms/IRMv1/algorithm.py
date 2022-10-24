import torch
from torch import Tensor
from typing import Callable, Union


NAME = "IRMv1"


def mean_nll(logits, y):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)


def penalty(logits, y, loss_fn: Callable):
    """Uses dummy linear classfier w of value 1.0.
    The loss and the grad are both "sizeless" tensors (scalar tensors).
    """
    scale = torch.tensor(1.).requires_grad_()
    loss = loss_fn(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


def compute_grads(batch_size: int,
                  loss_fn: Union[Callable, None],
                  n_envs: int,
                  model_params: list,
                  output: Tensor,
                  target: Tensor,
                  device: torch.device,
                  epoch: int,
                  **kwargs
                  ) -> tuple:
    """Sets gradient values using outputs, targets and loss_fn(outputs, targets).
        * optimizer.zero_grad() called before entering this function
        * optimizer.step() called right after this function returns
    :param epoch: epoch number
    :param kwargs: parameters:
        * irm_lambda: scaling factor IRM term in loss func
        * penalty_anneal_epochs: epoch number from which to start applying
    """
    ERM_term: float
    L2_reg_term: float
    IRM_term: float
    # determine parameters
    penalty_anneal_epochs = 1000000  # default value; shuld be overriden by a smaller one
    use_epoch_lambda_map = kwargs.get('use_epoch_lambda_map')
    if use_epoch_lambda_map:
        epoch_lambda_map = kwargs.get('epoch_lambda_map')
        if epoch_lambda_map is None:
            raise Exception('missing parameter')
        if epoch in epoch_lambda_map.keys():
            irm_lambda = epoch_lambda_map[epoch]
        else:
            max_defined_epoch = max(epoch_lambda_map.keys())
            irm_lambda = epoch_lambda_map[max_defined_epoch]
            penalty_anneal_epochs = 1000000  # high so it never hits the bound, though it should never happen
    else:
        irm_lambda = kwargs.get('irm_lambda')
        penalty_anneal_epochs = kwargs.get('penalty_anneal_epochs')
    l2_reg = kwargs.get('l2_reg')
    if irm_lambda is None or penalty_anneal_epochs is None or l2_reg is None:
        raise Exception('missing parameter')
    loss_fn = loss_fn if loss_fn else mean_nll

    outputs = output.view(n_envs, batch_size, -1)  # will iterate over each env (first dim)
    targets = target.view(n_envs, batch_size, -1)
    outputs = outputs.squeeze(-1)
    targets = targets.squeeze(-1)

    env_losses = [dict() for _ in range(n_envs)]
    for env_outputs, env_targets, losses in zip(outputs, targets, env_losses):
        losses['nll'] = loss_fn(env_outputs, env_targets)
        losses['penalty'] = penalty(env_outputs, env_targets, loss_fn)

    train_nll = torch.stack([env_loss['nll'] for env_loss in env_losses]).mean()
    train_penalty = torch.stack([env_loss['penalty'] for env_loss in env_losses]).mean()
    IRM_term = train_penalty.item()

    weight_norm = torch.tensor(0.).to(device=device)
    for w in model_params:
        weight_norm += w.norm().pow(2)

    # LOSS: nll ERM term
    loss: torch.Tensor = train_nll.clone()
    ERM_term = loss.item()
    # LOSS: L2 term
    l2_term = l2_reg * weight_norm
    loss += l2_term
    L2_reg_term = l2_term.item()

    # this uses the current epoch number to apply a penalty. For now just hardocde it
    #penalty_weight_applied = penalty_weight if step >= flags.penalty_anneal_iters else 1.0

    # LOSS: IRM term
    if use_epoch_lambda_map:
        loss += irm_lambda * train_penalty  # irm_lambda == penalty_weight
        loss /= irm_lambda
    else:
        if epoch > penalty_anneal_epochs:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= irm_lambda
            loss += train_penalty  # irm_lambda == penalty_weight
        else:
            loss += train_penalty

    loss.backward()
    return ERM_term, L2_reg_term, IRM_term


def build_lambda_map(intervals=None) -> dict:
    """this dict maps epoch number to irm_lambda value"""
    lambda_map = dict()
    if intervals is None:
        intervals = [
            ([1, 16], 1.0),
            ([16, 26], 1000.0),
            ([26, 36], 10000.0),
            ([36, 46], 30000.0),
            ([46, 66], 50000.0)
        ]
    for bounds_values in intervals:
        bounds = bounds_values[0]
        irm_lambda = bounds_values[1]
        lower, upper = bounds[0], bounds[1]
        for i in range(lower, upper):
            lambda_map[i] = irm_lambda
    return lambda_map
