import timeit
import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print(mps_device)
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

a_mps = torch.rand(1000, device='mps')
b_mps = torch.rand((1000, 1000), device='mps')
#c_mps = torch.rand((1000, 1000), device='cpu')

print('mps', timeit.timeit(lambda: torch.matmul(b_mps, b_mps), number=10_000))
#print('cpu', timeit.timeit(lambda: c_mps @ c_mps, number=10_000))
