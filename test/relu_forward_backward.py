import torch
import torch.nn as nn
import time

a = torch.randn(64, 64, 55, 55, requires_grad=True)
is_gpu_available = False
if torch.cuda.is_available():
    is_gpu_available = True

if is_gpu_available :
    a = a.cuda()

m = nn.ReLU()
output = m(a)
print ("Finished")

torch.cuda.synchronize()
start_time = time.time()
print ("Backward started")
output.sum().backward(retain_graph=True)
print ("Backward finished.")
torch.cuda.synchronize()
print ("Total time is {}".format(time.time() - start_time))


output_size = output.size()
grad_output = torch.randn(output_size[0], output_size[1], output_size[2], output_size[3])
if torch.cuda.is_available():
    grad_output = grad_output.cuda()

torch.cuda.synchronize()
val = time.time()
print ("Running backward")

for i in range(2):
    output.backward(grad_output, retain_graph=True)

print ("Backward finished.")
torch.cuda.synchronize()
print ("Totatl time is : {}".format(time.time() - val))
