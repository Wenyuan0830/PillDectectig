import torch.nn as nn


class HeatmapMSELoss(nn.Module):
    def __init__(self, target_weight=[0.25, 1.]):
        super(HeatmapMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.target_weight = target_weight

    def forward(self, output, target):
        batch_size = output.size(0)
        n_out_channels = output.size(1)

        heatmaps_pred = output.reshape((batch_size, n_out_channels, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, n_out_channels, -1)).split(1, 1)

        loss = 0
        for idx in range(n_out_channels):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.target_weight is not None:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(self.target_weight[idx]),
                    heatmap_gt.mul(self.target_weight[idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / n_out_channels


def build_loss(args):
    print('[INFO] Building loss function...')
    if args.loss_name == 'heatmap_mse':
        target_weight = [1. / args.class_ratio, 1.]
        loss = HeatmapMSELoss(target_weight=target_weight)
    else:
        raise NotImplementedError()

    return loss