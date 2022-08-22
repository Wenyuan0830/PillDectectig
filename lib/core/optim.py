import math
import torch


def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
            curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
            and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
                (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
                1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = curr_lr
    return curr_lr


def build_optimizer(args, model):
    print('[INFO] Building optimizer...')
    params_with_decay = []
    params_without_decay = []
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            continue
        if args.filter_biases_wd and (len(param.shape) == 1 or name.endswith("bias")):
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    if args.filter_biases_wd:
        param_groups = [
            {"params": params_without_decay, "weight_decay": 0.0},
            {"params": params_with_decay, "weight_decay": args.weight_decay},
        ]
    else:
        param_groups = [
            {"params": params_with_decay, "weight_decay": args.weight_decay},
        ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.base_lr)
    return optimizer
