import torch
from typing import Callable

if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "mps"
else:
    device = torch.device("cpu")
    device_name = "cpu"

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func