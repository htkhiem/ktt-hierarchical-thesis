import torch

def save_checkpoint(checkpoint, best_yet, path, best_path):
    """ Save checkpoint to PyTorch .pt format.
     If this is the best-yet checkpoint, also make a backup to another path.
    """
    torch.save(checkpoint, path)
    if best_yet:
        torch.save(checkpoint, best_path)


if __name__ == "__main__":
    pass
