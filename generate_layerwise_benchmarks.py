import os
import sys
import argparse
import random
import time

import torch
import torch.nn as nn
import torchvision

import pymi
from pymi import ModuleInstrumentation as mi

def run_layerwise_benchmark(network_name, batch_size, iterations, is_gpu_available):
    print ("INFO: Benchmark will be run with following details : ")
    print ("Network : {}, Batchsize : {}, Iterations : {}".format(network_name, batch_size, iterations))
    net = torchvision.models.alexnet()
    input_size = [1, 3, 224, 224]
    is_debug = True
    layer_timer = mi.PyModuleInstrumentation(net, input_size, iterations, is_debug)
    net_data = layer_timer.generate_layerwise_profile_info()
    print ("OK: Finished generating layerwise benchmark for network : {}".format(network_name))
    layer_timer.generate_statistics(net_data)

def main():
    network_name = args.network
    batch_size = args.batch_size
    iterations = args.iterations
    is_gpu_available = False
    if torch.cuda.is_available():
        print ("INFO: GPU is available for running benchmark, switching to use GPU")
        is_gpu_available = True
    else:
        print ("INFO: GPU not available, running using CPU.")
    
    run_layerwise_benchmark(network_name, batch_size, iterations, is_gpu_available)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, required=True, default='resnet50', help="Torchvision network name")
    parser.add_argument('--batch-size', type=int, required=False, default=64, help="Batchsize of the model.")
    parser.add_argument('--iterations', type=int, required=False, default=10, help="Number of iterations to run;")

    args = parser.parse_args()

    main()

