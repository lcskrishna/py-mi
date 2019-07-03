# py-mi (A Pytorch Module Instrumentor)
py-mi is a tool on top of Pytorch module (nn.Module). This tool collects the necessary statistics related to a Module created using pytorch.
Also, this module doesn't use any gpu related profiling tools underneath like nvprof to collect the information. 
This can be used on any GPU or CPU that supports pytorch.

Note : Currently, this tool only supports FP32 computation.

## Contents

* [Pre-requisites](#pre-requisites)
* [Getting Started](#getting-started)
* [Usage](#how-to-use-this-tool-in-your-python-scripts-)
* [Demo](#to-run-a-small-demo-run-the-following-example)

## Pre-requisites:

Make sure to have the following installed.
* python 2.7 or python 3.6
* pytorch 
* torchvision

To install pytorch and torchvision follow the below links:
1. pytorch :[Official pytorch site](https://github.com/pytorch/pytorch) or [Pytorch on ROCM](https://github.com/ROCmSoftwarePlatform/pytorch/wiki/Building-PyTorch-for-ROCm)
2. torchvision : [vision](https://github.com/pytorch/vision)

Or use the pre-installed pytorch dockers available from their sites respectively.

## Getting Started

To get started, first install the pymi module into your workspace.

**To install:**
Clone the repository and run the following command inside the source.

```
python setup.py install
```

This will install a python module named pymi.

## How to use this tool in your python scripts :

Import the following in your python scripts that you want to do get some analytics.
```
import pymi
from pymi import ModuleInstrumentation as mi
```

To get the layerwise timings profile for a particular network use the following:
```
mi.PyModuleInstrumentation(net, input_size, iterations, is_debug).generate_layerwise_profile_info()
```
In the above command,
* net is the network created using nn.Module, for example torchvision.models.alexnet()
* input_size is an array of input size like ([1,3,224,224])
* iterations is the number of iterations you wish to profile.
* is_debug is a flag to show debug mesasges (True/False)

## To run a small demo, run the following example

```
python generate_layerwise_benchmarks.py --network alexnet
```

