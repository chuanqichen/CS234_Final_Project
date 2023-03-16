import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "mps"
else:
    device = torch.device("cpu")
    device_name = "cpu"
