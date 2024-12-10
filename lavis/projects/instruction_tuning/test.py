import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print("GPU Name:", gpu_name)
else:
    print("No GPU available")
