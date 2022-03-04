import logging
import torch
from torch import Tensor


def config_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)


def get_window_positions(n: int, window: int) -> Tensor:
    """
    Create a tensor (n, window) such that the i-th row has the indices of the
    columns attended by it. E.g.,

    [[2, 0, 1],  # shifts as padding
     [0, 1, 2],
     [1, 2, 3],
     [2, 3, 4],
     ...]
    """
    half_window = window // 2

    # form indices for columns in an attention window pattern
    # e.g. [0, 1, 2], [1, 2, 3], [2, 3, 4] etc
    r = torch.arange(n).view(-1, 1)
    attended = r + torch.arange(-half_window, half_window + 1)

    # make the windows at the first and last few words the same size as in
    # the middle
    attended[attended < 0] += window
    attended[attended >= n] -= window

    return attended


def get_length_mask(lengths: Tensor, max_length: int = None) -> Tensor:
    """Create a (batch_size, max_length) boolean mask
    True for true positions and False for padding"""
    if max_length is None:
        max_length = lengths.max()
    r = torch.arange(max_length).unsqueeze(0).to(lengths.device)
    mask = r < lengths.unsqueeze(-1)
    return mask