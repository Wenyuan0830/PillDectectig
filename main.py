import argparse
import os
import time

from torch.utils.data import DataLoader

import torch

from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from lib.core.loss import build_loss
from lib.core.optim import build_optimizer, adjust_learning_rate
from lib.model.resnet_fcn import ResNetFCN
from lib.dataset.generators import StackedHeatmapGenerator
from lib.dataset.heatmap_dataset import DefaultHeatmapDataset
from lib.utils.io import save_checkpoint, resume_if_possible
from lib.utils.misc import SmoothedValue


def make_args_parser():
    parser = argparse.ArgumentParser('Training', add_help=False)

    ##### Optimizer #####
    parser.add_argument('--base_lr', default=5e-4, type=float)
    parser.add_argument('--warm_lr', default=1e-6, type=float)
    parser.add_argument('--warm_lr_epochs', default=9, type=int)
    parser.add_argument('--final_lr', default=1e-6, type=float)
    parser.add_argument('--lr_scheduler', default='cosine', type=str)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--filter_biases_wd', default=False, action='store_true')
    parser.add_argument(
        '--clip_gradient', default=0.1, type=float, help='Max L2 norm of the gradient'
    )

    ##### Model #####
    parser.add_argument(
        '--model_name',
        default='fcn_resnet50',
        type=str,
        help='Name of the model',
        choices=['fcn_resnet50', 'fcn_resnet101'],
    )

    parser.add_argument('--output_size', default=128, type=int)
    parser.add_argument('--n_out_channels', default=2, type=int)
    parser.add_argument('--pretrained', default=True, action='store_true')
    parser.add_argument('--use_pretrain_head', default=True, action='store_true')
    parser.add_argument('--use_aux', default=False, action='store_false')

    ##### Loss #####
    parser.add_argument(
        '--loss_name',
        default='heatmap_mse',
        type=str,
        help='Choice of the loss',
        choices=['heatmap_mse']
    )
    parser.add_argument(
        '--class_ratio',
        default=4,
        type=int,
        help='present/missing ratio')

    ##### Dataset #####
    parser.add_argument(
        '--dataset_name',
        default='default',
        type=str,
        help='Choice of dataset',
        choices=['default'],
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='./public_data',
        help='Root directory containing the dataset files.'
    )
    parser.add_argument('--dataset_num_workers', default=8, type=int)
    parser.add_argument('--batchsize_per_gpu', default=32, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--heatmap_size', default=128, type=int)
    parser.add_argument('--shuffle', default=True, action='store_true')
    parser.add_argument(
        '--heatmap_generator',
        default='default',
        type=str,
        help='Choice of heatmap generator',
        choices=['default'],
    )

    ##### Training #####
    parser.add_argument('--start_epoch', default=-1, type=int)
    parser.add_argument('--max_epoch', default=300, type=int)
    parser.add_argument('--eval_every_epoch', default=10, type=int)
    parser.add_argument('--seed', default=0, type=int)

    ##### I/O #####
    parser.add_argument('--checkpoint_dir', default='./ckpt', type=str)
    parser.add_argument('--log_every', default=10, type=int)
    parser.add_argument('--log_metrics_every', default=20, type=int)
    parser.add_argument('--save_separate_checkpoint_every_epoch', default=100, type=int)

    return parser


def train_one_epoch(args, curr_epoch, model, optimizer, criterion, dataset_config, dataset_loader):
    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    model.train()

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    for batch_idx, batch_data in enumerate(dataset_loader):
        curr_time = time.time()
        # curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data:
            # TODO: convert data to tensor!
            if not key.startswith('_'):  # all keys start with _ is not tensor
                batch_data[key] = batch_data[key].to(net_device)

        # Forward pass
        optimizer.zero_grad()
        ipt = batch_data['input']
        heatmap_gt = batch_data['heatmap']
        outputs = model(ipt)
        heatmap_pred = outputs['out']  # [batch, 2, H, W]
        heatmap_aux_pred = outputs['aux']

        # Compute loss
        loss = criterion(heatmap_pred, heatmap_gt)

        loss.backward()

        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()

        time_delta.update(time.time() - curr_time)
        loss_avg.update(loss.item())

    return loss_avg.avg


def visualize(model, dataset_loader, filename='result.jpg'):
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataset_loader):
            break

        for key in batch_data:
            # TODO: convert data to tensor!
            if not key.startswith('_'):  # all keys start with _ is not tensor
                batch_data[key] = batch_data[key].to('cuda')

        # Forward pass
        ipt = batch_data['input']
        heatmap_gt = batch_data['heatmap']
        outputs = model(ipt)
        heatmap_pred = outputs['out']

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
        axs = axs.flatten()
        plt.suptitle(batch_data['_id'])

        plt.title('gt ch0')
        plt.sca(axs[0])
        plt.imshow(heatmap_gt[0][0].detach().cpu().numpy().tolist())

        plt.title('gt ch1')
        plt.sca(axs[1])
        plt.imshow(heatmap_gt[0][1].detach().cpu().numpy().tolist())

        plt.title('pd ch0')
        plt.sca(axs[2])
        plt.imshow(heatmap_pred[0][0].detach().cpu().numpy().tolist())

        plt.title('pd ch1')
        plt.sca(axs[3])
        plt.imshow(heatmap_pred[0][1].detach().cpu().numpy().tolist())

        plt.savefig(filename)


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # build dataset
    if args.dataset_name == 'default':
        dataset_cls = DefaultHeatmapDataset
    else:
        raise NotImplementedError
    if args.heatmap_generator == 'default':
        hm_generator_cls = StackedHeatmapGenerator
    else:
        raise NotImplementedError

    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    ds = dataset_cls(
        args.dataset_path,
        heatmap_generator=hm_generator_cls(
            args.heatmap_size,
            n_hms=args.n_out_channels,
            rescale_factor=args.heatmap_size / args.image_size
        ),
        transform=img_transform,
        args=args
    )

    if args.shuffle:
        sampler = torch.utils.data.RandomSampler(ds)
    else:
        sampler = torch.utils.data.SequentialSampler(ds)

    dataloaders = {
        'train': DataLoader(
            ds,
            sampler=sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
        ),
        'visual': DataLoader(
            ds,
            sampler=sampler,
            batch_size=1,
            num_workers=args.dataset_num_workers,
        )
    }

    # build model
    model = ResNetFCN(
        base_model_name=args.model_name,
        output_size=args.heatmap_size,
        n_out_channels=args.n_out_channels,
        pretrained=args.pretrained,
        use_pretrain_head=args.use_pretrain_head,
        use_aux=args.use_aux
    )

    if torch.cuda.is_available():
        model = model.cuda()

    # build criterion
    criterion = build_loss(args)

    # build optimizer
    optimizer = build_optimizer(args, model)

    # resume if possible
    last_checkpoint = os.path.join(args.checkpoint_dir, 'checkpoint.pth')
    if not os.path.isdir(args.checkpoint_dir) or not os.path.isfile(last_checkpoint):
        saved_epoch = -1
        best_val_metrics = {
            'heatmap_mse': float('inf')
        }
    else:
        saved_epoch, best_val_metrics = resume_if_possible(args.checkpoint_dir, model, optimizer)
    print(best_val_metrics)
    args.start_epoch = saved_epoch + 1

    # TODO: placeholder variable
    dataset_config = {}
    # do train
    for epoch in range(args.start_epoch, args.max_epoch):
        current_loss = train_one_epoch(args, epoch, model, optimizer, criterion, dataset_config, dataloaders['train'])

        if current_loss < best_val_metrics['heatmap_mse']:
            best_val_metrics['heatmap_mse'] = current_loss
            # save the latest best model
            save_checkpoint(
                args.checkpoint_dir,
                model,
                optimizer,
                epoch,
                args,
                best_val_metrics,
                filename="checkpoint_best.pth",
            )

        # save by the end of a training epoch
        save_checkpoint(
            args.checkpoint_dir,
            model,
            optimizer,
            epoch,
            args,
            best_val_metrics,
            filename=f"checkpoint_{epoch}.pth",
        )

        print("==" * 10)
        print(f"Epoch [{epoch}/{args.max_epoch}]; Loss: {current_loss}")
        # print("==" * 10)

        if epoch % args.eval_every_epoch == 0:
            print('[INFO] Saving visual result...')
            visualize(model, dataset_loader=dataloaders['visual'], filename=f'{args.checkpoint_dir}/epoch_{epoch}.jpg')


if __name__ == '__main__':
    parser = make_args_parser()
    args = parser.parse_args()
    main(args)
