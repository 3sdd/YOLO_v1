import torch.nn as nn

def conv_bn_relu(in_channels,out_channles,kernel_size,stride,padding):
    return [
        nn.Conv2d(in_channels,out_channles,kernel_size=kernel_size,stride=stride,padding=padding), 
        nn.BatchNorm2d(out_channles),
        nn.LeakyReLU(0.1,inplace=True),
    ]