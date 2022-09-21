import torch
from torch import Tensor
from typing import Callable, Union

l2_regularizer_weight = 0.001


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


def compute_grads(irm_lambda: float,
                  batch_size: int,
                  loss_fn: Union[Callable, None],
                  n_envs: int,
                  model_params: list,
                  output: Tensor,
                  target: Tensor,
                  device: torch.device
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

    train_nll = torch.stack([env_loss['nll'] for env_loss in env_losses]).mean()
    train_penalty = torch.stack([env_loss['penalty'] for env_loss in env_losses]).mean()

    weight_norm = torch.tensor(0.).to(device=device)
    for w in model_params:
        weight_norm += w.norm().pow(2)

    # LOSS: nll ERM term
    loss: torch.Tensor = train_nll.clone()
    # LOSS: L2 term
    loss += l2_regularizer_weight * weight_norm

    # this uses the current epoch number to apply a penalty. For now just hardocde it
    #penalty_weight_applied = penalty_weight if step >= flags.penalty_anneal_iters else 1.0

    # LOSS: IRM term
    loss += irm_lambda * train_penalty  # irm_lambda == penalty_weight
    if irm_lambda > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
        loss /= irm_lambda

    loss.backward()
    return
