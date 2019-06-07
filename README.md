# py-mi (A Pytorch Module Instrumentor)
py-mi is a tool on top of Pytorch module (nn.Module). This tool collects the necessary statistics related to a Module created using pytorch.
Currently, This module generates layerwise profiling information without using any specific gpu related tools. 

Also, this module doesn't use any gpu related profiling tools underneath like nvprof to collect the information. 
This can be used on any GPU or CPU that supports pytorch.

## Pre-requisites:

Make sure to have the following installed.
* python 2.7 or python 3.6
* pytorch 
* torchvision

To install pytorch and torchvision follow the below links:
1. pytorch :[Official pytorch site](https://github.com/pytorch/pytorch) or [Pytorch on ROCM](https://github.com/ROCmSoftwarePlatform/pytorch/wiki/Building-PyTorch-for-ROCm)
2. torchvision : [vision](https://github.com/pytorch/vision)

Or use the pre-installed pytorch dockers. 

## Getting Started

To get started, first install the pymi module into your workspace.

**To install:**
Clone the repository and run the following command inside the source.

```
python setup.py install
```

This will install a python module named pymi.

## How to use this tool in your python scripts :'

Coming soon .. 
