import torch
import torch.nn as nn

a = torch.randn(1, 3, 224, 224, requires_grad=True)
if torch.cuda.is_available():
    a = a.cuda()

m = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)).cuda()

print("Running forward")
output = m(a)
output_size = output.size()
print ("Forward finished.")

grad_output = torch.randn(output_size[0], output_size[1], output_size[2], output_size[3])
if torch.cuda.is_available():
    grad_output = grad_output.cuda()

print ("Running backward")

output.backward(grad_output)

print ("Backward finished.")
