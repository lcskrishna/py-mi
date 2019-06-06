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
        inp = torch.randn(self.input_size[0], self.input_size[1], self.input_size[2], self.input_size[3], requires_grad=True)
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

    ### Note: This method is created to avoid Leaf variable was used in an inplace operation issue
    ## Idea is leaf variable (i.e. variable /tensor created directly) can't be used for inplace operation.
    def get_intermediate_input(self, input_size):
        if len(input_size) == 4:
            a = torch.randn(input_size[0], input_size[1], input_size[2], input_size[3], requires_grad=True)
            a = a.clone() 
            if self.gpu_available :
                a = a.cuda()
            return a

        if len(input_size) == 3:
            a = torch.randn(input_size[0], input_size[1], input_size[2], requires_grad=True)
            a = a.clone()
            if self.gpu_available :
                a = a.cuda()
            return a
    
        if len(input_size) == 2:
            a = torch.randn(input_size[0], input_size[1], requires_grad=True)
            a = a.clone()
            if self.gpu_available:
                 a = a.cuda()
            return a

    def get_grad_output(self, input_size):
        if len(input_size) == 4:
            a = torch.randn(input_size[0], input_size[1], input_size[2], input_size[3], requires_grad=True)
            if self.gpu_available :
                a = a.cuda()
            return a

        if len(input_size) == 3:
            a = torch.randn(input_size[0], input_size[1], input_size[2], requires_grad=True)
            if self.gpu_available :
                a = a.cuda()
            return a
    
        if len(input_size) == 2:
            a = torch.randn(input_size[0], input_size[1], requires_grad=True)
            if self.gpu_available:
                 a = a.cuda()
            return a

    def compute_backward_differently(self, layer):
        is_different = False
        if isinstance(layer, nn.ReLU):
            is_different=True
        elif isinstance(layer, nn.MaxPool2d):
            is_different = True
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            is_different = True
        elif isinstance(layer, nn.Dropout):
            is_different = True

        return is_different

    def generate_time(self, layer, x, iter=10):
        if self.gpu_available:
            x = x.cuda()
            layer = layer.cuda()
        
        output_wm1 = layer(x)
        output_wm2 = layer(x)
       
        print ("INFO: Running forward .. ")
        torch.cuda.synchronize()
        start_time = time.time()
        for j in range(iter):
            output = layer(x)
        torch.cuda.synchronize()
        forward_time = time.time() - start_time
        output_size = output_wm2.size()
        
        backward_time = 0
        if self.compute_backward_differently(layer):
            m = nn.Sequential(layer)
            if self.gpu_available:
                m = m.cuda()
            x_size = x.size()
            a = self.get_intermediate_input(x_size)
            output = m(a)

            print ("INFO: Running backward warmup")
            torch.cuda.synchronize()
            start_time_bk_w = time.time()
            output.sum().backward(retain_graph=True)
            torch.cuda.synchronize()
            print ("OK: Backward warmup finished.")

            o_size = output.size()
            grad_o_new = self.get_grad_output(o_size)

            print ("INFO: Running backward")
            torch.cuda.synchronize()
            start_time_bk = time.time()
            for i in range(iter):
                output.backward(grad_o_new, retain_graph=True)
            print ("OK: Backward path finished.")
            backward_time = time.time() - start_time_bk

        else:
            grad_output = self.get_grad_output(output_size)
            print ("INFO: Running backward warmup.")
            output_wm2.backward(grad_output, retain_graph=True)
            print ("INFO: Backward warmup finished.")

            print ("INFO: Running backward..")
            torch.cuda.synchronize()
            start_time_bk = time.time()
            for j in range(iter):
                output_wm2.backward(grad_output, retain_graph=True)
            torch.cuda.synchronize()
            print ("OK: Backward path finished.")
            backward_time = time.time() - start_time_bk 
            
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
            print ("Input size is : {}".format(x.size()))
            forward_time , backward_time, output_size = self.generate_time(layer, x)
            print ("Ouptut size is : {}".format(output_size))
            layer_data['forward_time'] = forward_time
            layer_data['backward_time'] = backward_time
            layer_data['input_size'] = x.size()
            layer_data['output_size'] = output_size
            net_layer_data[i] = layer_data
            print ("Forward Time is {}".format(forward_time))
            print ("Backward Time is : {}".format(backward_time))

        return net_layer_data
