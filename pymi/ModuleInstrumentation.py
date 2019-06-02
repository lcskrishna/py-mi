import torch
import torch.nn as nn

import os
import sys
import time
import random

class PyModuleInstrumentation():
    def __init__(self, network, input_size, iterations, is_debug):
        self.net = network
        self.input_size = input_size
        self.iterations = iterations
        self.gpu_available = self.is_gpu_available()
        self.debug_enabled = is_debug

    def is_gpu_available(self):
        is_gpu_available = False
        if torch.cuda.is_available():
            print ("INFO: GPU is available for computation, switching to gpu")
            is_gpu_available = True
        else:
            print ("INFO: GPU is not available, hence using CPU")

        return is_gpu_available

    def get_input(self):
        inp = torch.randn(self.input_size[0], self.input_size[1], self.input_size[2], self.input_size[3])
        if self.gpu_available :
            inp = inp.cuda()
        return inp

    def get_network(self):
        if self.gpu_available:
            net = net.cuda()
        return net

    def getLayers(self, module, layer_info):
        sub_modules = module.__dict__['_modules']
        for name, sub_module in sub_modules.items():
            if sub_module is None or isinstance(sub_module, nn.Module) is False:
                break
            if isinstance(sub_module, nn.Container) or isinstance(sub_module, nn.Sequential):
                self.getLayers(sub_module, layer_info)
            else:
                layer_info.append(sub_module)
                if self.debug_enabled :
                    print (sub_module.__class__)

    def get_layer_info(self):
        layer_info = []
        self.getLayers(self.net, layer_info)
        return layer_info

    def generate_layerwise_profile_info(self):
        '''
            1. Finish implementation of get linear info.
        '''
        x = self.get_input()
        layer_info = self.get_layer_info()
        
        output_sizes = {}
        layer_data = {}
        net_layer_data = {}
        num_linear_layer = 0

        print ("ERROR: Need to finish the timeline generation")
        sys.exit(1) 
            
