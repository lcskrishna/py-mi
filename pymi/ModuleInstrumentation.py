import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable

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
            if isinstance(sub_module, nn.Container) or isinstance(sub_module, nn.Sequential) or isinstance(sub_module, torchvision.models.resnet.Bottleneck):
                self.getLayers(sub_module, layer_info)
            else:
                layer_info.append(sub_module)
                if self.debug_enabled :
                    print (sub_module.__class__)

    def get_layer_info(self):
        layer_info = []
        self.getLayers(self.net, layer_info)
        return layer_info

    def generate_time(self, layer, x, iter=10):
        #inp = Variable(x, requires_grad=True)
        inp = x
        if self.gpu_available:
            inp = inp.cuda()
            layer = layer.cuda()
        output_warmup1 = layer(inp)
        output_warmup2 = layer(inp)
        output_size = output_warmup1.size()

        ## Forward time computation.
        torch.cuda.synchronize()
        start_time = time.time()
        for j in range(iter):
            output = layer(inp)
        torch.cuda.synchronize()
        forward_time = time.time() - start_time
        print ("DEBUG: Finished forward computation")
       
        #grad_output = output_warmup2.clone().normal_()
        #grad_output = torch.randn(output_size[0], output_size[1], output_size[2], output_size[3], requires_grad=True)
        #if self.gpu_available:
        #    grad_output = grad_output.cuda()
        #print ("INFO: computing backward path.")
        #if isinstance(layer, nn.ReLU):
        #    print ("RELU Backward")
        #    #output_warmup2.backward(grad_output, retain_graph=True)
        #    output_warmup2.sum().backward()
        #    print ("RELU backward finished.")
        #output_warmup2.backward(grad_output, retain_graph=True)
        ### Backward time computation.
        #torch.cuda.synchronize()
        #start_time_back = time.time()
        #for j in range(iter):
        #    output_warmup2.backward(grad_output, retain_graph=True)
        #torch.cuda.synchronize()
        #backward_time = time.time() - start_time_back
        #
        #print ("DEBUG: Finished backward computation.") 
        backward_time = 0
        return forward_time, backward_time, output_size
        
    def generate_layerwise_profile_info(self):
        
        x = self.get_input()
        print (x.size())
        layer_info = self.get_layer_info()
        
        output_sizes = {}
        layer_data = {}
        net_layer_data = {}
        num_linear_layer = 0

        for i in range(len(layer_info)):
            layer_data['layer_num'] = i
            layer = layer_info[i]
            if i != 0:
                prev_layer_info = net_layer_data[i - 1]
                prev_output_size = prev_layer_info['output_size']
                
                if (len(prev_output_size) == 4):
                    x = torch.randn(prev_output_size[0], prev_output_size[1], prev_output_size[2], prev_output_size[3])
                    if self.gpu_available:
                        x = x.cuda()
                elif (len(prev_output_size) == 3):
                    x = torch.randn(prev_output_size[0], prev_output_size[1], prev_output_size[2])
                    if self.gpu_available:
                        x = x.cuda()
                elif (len(prev_output_size) == 2):
                    x = torch.randn(prev_output_size[0], prev_output_size[1])
                    if self.gpu_available:
                        x = x.cuda()
                if (isinstance(layer, nn.Linear) and num_linear_layer == 0):
                    if len(prev_output_size) == 4:
                        x = x.view(-1, prev_output_size[1] * prev_output_size[2] * prev_output_size[3])
                        if self.gpu_available:
                            x = x.cuda()
                    if len(prev_output_size) == 3:
                        x = x.view(-1,  prev_output_size[1] * prev_output_size[2])
                        if self.gpu_available:
                            x = x.cuda()
                    if len(prev_output_size) == 2:
                        x = x.view(-1, prev_output_size[1])
                        if self.gpu_available:
                            x = x.cuda()
                    num_linear_layer = num_linear_layer + 1
            
            print ("------------------------ Layer num {} ---------------------- ".format(i))
            print (layer)
            forward_time , backward_time, output_size = self.generate_time(layer, x)
            layer_data['forward_time'] = forward_time
            layer_data['backward_time'] = backward_time
            layer_data['input_size'] = x.size()
            layer_data['output_size'] = output_size
            net_layer_data[i] = layer_data
            print ("Forward Time is {}".format(forward_time))
            print ("Backward Time is : {}".format(backward_time))

        return net_layer_data
