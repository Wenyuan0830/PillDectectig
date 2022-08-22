import torch
import os


def is_primary():
    return torch.distributed.get_rank() == 0


def save_checkpoint(
        checkpoint_dir,
        model,
        optimizer,
        epoch,
        args,
        best_val_metrics,
        filename=None,
):
    if filename is None:
        filename = f'checkpoint_{epoch:04d}.pth'
    checkpoint_name = os.path.join(checkpoint_dir, filename)

    state_data = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
        'best_val_metrics': best_val_metrics,
    }
    torch.save(state_data, checkpoint_name)


def resume_if_possible(checkpoint_dir, model, optimizer):
    """ Resume if checkpoint is available. """
    epoch = -1
    best_val_metrics = {}

    last_checkpoint = os.path.join(checkpoint_dir, 'checkpoint.pth')

    state_data = torch.load(last_checkpoint)
    epoch = state_data['epoch']
    best_val_metrics = state_data['best_val_metrics']
    print(f'Found checkpoint at {epoch}. Resuming.')

    model.load_state_dict(state_data['model'])
    optimizer.load_state_dict(state_data['optimizer'])
    print(
        f'Loaded model and optimizer state at {epoch}. Loaded best val metrics so far.'
    )
    return epoch, best_val_metrics
