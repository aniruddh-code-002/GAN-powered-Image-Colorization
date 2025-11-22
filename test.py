import torch
print(torch.cuda.get_device_name(0))
print(torch.rand(10000, 10000, device='cuda') @ torch.rand(10000, 10000, device='cuda'))
