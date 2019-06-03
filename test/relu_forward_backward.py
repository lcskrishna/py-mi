import torch
import torch.nn as nn

a = torch.randn(2, 64, 55, 55, requires_grad=True)
is_gpu_available = False
if torch.cuda.is_available():
    is_gpu_available = True

if is_gpu_available :
    a = a.cuda()

m = nn.ReLU()
output = m(a)
print ("Finished")

print ("Backward started")
output.sum().backward()
print ("Backward finished.")



