import torch
import torch.nn as nn

from .util import conv_bn_relu

class TinyYOLOv1(nn.Module):
    def __init__(self,num_classes,S,B):
        super().__init__()
        self.num_classes=num_classes
        self.S=S
        self.B=B

        
        self.model=nn.Sequential(
            # (1,3,448,448) -> ???
            # *conv_bn_relu(3,16,kernel_size=7,stride=2,padding=7//2), #TODO:pad=1
            *conv_bn_relu(3,16,kernel_size=3,stride=2,padding=1), #TODO:pad=1
            nn.MaxPool2d(kernel_size=2,stride=2),

            #(1,???) -> (1, ????)
            # *conv_bn_relu(16,32,kernel_size=3,stride=1,padding=3//2),#TODO:pad=1
            *conv_bn_relu(16,32,kernel_size=3,stride=1,padding=1),#TODO:pad=1
            nn.MaxPool2d(2,stride=2),

            #(1,) -> (1,)
            # *conv_bn_relu(32,64,kernel_size=3,stride=1,padding=3//2), #TODO:pad=1
            *conv_bn_relu(32,64,kernel_size=3,stride=1,padding=1), #TODO:pad=1
            nn.MaxPool2d(2,stride=2),

            # *conv_bn_relu(64,128,kernel_size=3,stride=1,padding=3//2), #TODO pad=1
            *conv_bn_relu(64,128,kernel_size=3,stride=1,padding=1), #TODO pad=1
            nn.MaxPool2d(2,stride=2),

            # *conv_bn_relu(128,256,kernel_size=3,stride=1,padding=3//2), #TODO:pad=1
            *conv_bn_relu(128,256,kernel_size=3,stride=1,padding=1), #TODO:pad=1
            nn.MaxPool2d(2,stride=2),

            #
            # *conv_bn_relu(256,512,kernel_size=3,stride=1,padding=3//2), #TODO pad=1
            *conv_bn_relu(256,512,kernel_size=3,stride=1,padding=1), #TODO pad=1
            nn.MaxPool2d(2,2),    

            # *conv_bn_relu(512,1024,kernel_size=3,stride=1,padding=3//2),#pad=1
            *conv_bn_relu(512,1024,kernel_size=3,stride=1,padding=1),#pad=1

            # *conv_bn_relu(1024,256,kernel_size=3,stride=1,padding=3//2), #TODO:pad=1
            *conv_bn_relu(1024,256,kernel_size=3,stride=1,padding=1), #TODO:pad=1
        )

        #1番目のLinear　と　Local Layerの違い？
        # (1,7*7*1024) -> (1,1470)
        self.fc_layers=nn.Sequential(
            # # nn.Dropout(0.5),
            nn.Linear(256*3*3,self.S*self.S*(self.B*5+self.num_classes)), 
        )

        self.sigmoid=nn.Sigmoid()
        


    def forward(self,x):
        out=self.model(x)
        out=out.view(out.size(0),-1) #flatten
        out=self.fc_layers(out)
        #(channel,height,width)
        out=out.view(-1,5*self.B+self.num_classes,self.S,self.S)
        out=self.sigmoid(out)
        #TODO: 全部にsigmoidかけてるけど、class_probabilityはsigmoidかけなくてもいいのでは？？？
        boxes,class_probability_map=torch.split(out,[5*self.B,self.num_classes],dim=1)
        return boxes,class_probability_map