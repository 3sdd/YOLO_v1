import torch
import torch.nn as nn
from .util import conv_bn_relu

class YOLOv1(nn.Module):
    def __init__(self,num_classes,S,B):
        super().__init__()
        self.num_classes=num_classes
        self.S=S
        self.B=B

        #padding が論文で書かれてない？
        #padding=kernrl_size//2
        #論文の図とサイズがあわない？

        
        self.model=nn.Sequential(
            # (1,3,448,448) -> (1,64,112,112)
            # nn.Conv2d(3,64,kernel_size=7,stride=2,padding=7//2), 
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(3,64,kernel_size=7,stride=2,padding=7//2),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #(1,64,112,112) -> (1,192,56,56)
            # nn.Conv2d(64,192,kernel_size=3,stride=1,padding=3//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(64,192,kernel_size=3,stride=1,padding=3//2),
            nn.MaxPool2d(2,stride=2),

            #(1,192,56,56) -> (1,512,28,28)
            # nn.Conv2d(192,128,kernel_size=1,stride=1,padding=1//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(192,128,kernel_size=1,stride=1,padding=1//2),

            # nn.Conv2d(128,256,kernel_size=3,stride=1,padding=3//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(128,256,kernel_size=3,stride=1,padding=3//2),

            # nn.Conv2d(256,256,kernel_size=1,stride=1,padding=1//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(256,256,kernel_size=1,stride=1,padding=1//2),
            
            # nn.Conv2d(256,512,kernel_size=3,stride=1,padding=3//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(256,512,kernel_size=3,stride=1,padding=3//2),

            nn.MaxPool2d(2,2),
            
            #(1,512,28,28) -> (1,1024,14,14)
            # nn.Conv2d(512,256,kernel_size=1,stride=1,padding=1//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(512,256,kernel_size=1,stride=1,padding=1//2),
            
            # nn.Conv2d(256,512,kernel_size=3,stride=1,padding=3//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(256,512,kernel_size=3,stride=1,padding=3//2),

            # nn.Conv2d(512,256,kernel_size=1,stride=1,padding=1//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(512,256,kernel_size=1,stride=1,padding=1//2),

            # nn.Conv2d(256,512,kernel_size=3,stride=1,padding=3//2),
            # nn.LeakyReLU(0.1,inplace=True),  
            *conv_bn_relu(256,512,kernel_size=3,stride=1,padding=3//2),

            # nn.Conv2d(512,256,kernel_size=1,stride=1,padding=1//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(512,256,kernel_size=1,stride=1,padding=1//2),

            # nn.Conv2d(256,512,kernel_size=3,stride=1,padding=3//2),
            # nn.LeakyReLU(0.1,inplace=True),   
            *conv_bn_relu(256,512,kernel_size=3,stride=1,padding=3//2),

            # nn.Conv2d(512,256,kernel_size=1,stride=1,padding=1//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(512,256,kernel_size=1,stride=1,padding=1//2),

            # nn.Conv2d(256,512,kernel_size=3,stride=1,padding=3//2),
            # nn.LeakyReLU(0.1,inplace=True),    
            *conv_bn_relu(256,512,kernel_size=3,stride=1,padding=3//2),

            # nn.Conv2d(512,512,kernel_size=1,stride=1,padding=1//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(512,512,kernel_size=1,stride=1,padding=1//2),

            # nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=3//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(512,1024,kernel_size=3,stride=1,padding=3//2),

            nn.MaxPool2d(2,2),

            # (1,1024,14,14) -> (1,1024,7,7)
            # nn.Conv2d(1024,512,kernel_size=1,stride=1,padding=1//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(1024,512,kernel_size=1,stride=1,padding=1//2),

            # nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=3//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(512,1024,kernel_size=3,stride=1,padding=3//2),

            # nn.Conv2d(1024,512,kernel_size=1,stride=1,padding=1//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(1024,512,kernel_size=1,stride=1,padding=1//2),

            # nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=3//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(512,1024,kernel_size=3,stride=1,padding=3//2),



            # (1,1024,7,7) -> (1,1024,7,7)
            # nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=3//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(1024,1024,kernel_size=3,stride=1,padding=3//2),

            # nn.Conv2d(1024,1024,kernel_size=3,stride=2,padding=3//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(1024,1024,kernel_size=3,stride=2,padding=3//2),

            # nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=3//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(1024,1024,kernel_size=3,stride=1,padding=3//2),

            # nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=3//2),
            # nn.LeakyReLU(0.1,inplace=True),
            *conv_bn_relu(1024,1024,kernel_size=3,stride=1,padding=3//2),

        )

        #1番目のLinear　と　Local Layerの違い？
        # (1,7*7*1024) -> (1,1470)
        self.fc_layers=nn.Sequential(
            nn.Linear(1024*7*7,4096), 
            nn.LeakyReLU(0.1,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(4096,self.S*self.S*(self.B*5+self.num_classes)),
            # nn.Sigmoid(),
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

# class YOLOv1(nn.Module):
#     def __init__(self,num_classes,S,B):
#         super().__init__()
#         self.S=S # S
#         self.num_classes=num_classes
#         self.B=B

#         #padding が論文で書かれてない？
#         #padding=kernrl_size//2
#         #論文の図とサイズがあわない？

        
#         self.model=nn.Sequential(
#             # (1,3,448,448) -> (1,64,112,112)
#             nn.Conv2d(3,64,kernel_size=7,stride=2,padding=7//2), 
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.MaxPool2d(kernel_size=2,stride=2),

#             #(1,64,112,112) -> (1,192,56,56)
#             nn.Conv2d(64,192,kernel_size=3,stride=1,padding=3//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.MaxPool2d(2,stride=2),

#             #(1,192,56,56) -> (1,512,28,28)
#             nn.Conv2d(192,128,kernel_size=1,stride=1,padding=1//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(128,256,kernel_size=3,stride=1,padding=3//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(256,256,kernel_size=1,stride=1,padding=1//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(256,512,kernel_size=3,stride=1,padding=3//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.MaxPool2d(2,2),
            
#             #(1,512,28,28) -> (1,1024,14,14)
#             nn.Conv2d(512,256,kernel_size=1,stride=1,padding=1//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(256,512,kernel_size=3,stride=1,padding=3//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(512,256,kernel_size=1,stride=1,padding=1//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(256,512,kernel_size=3,stride=1,padding=3//2),
#             nn.LeakyReLU(0.1,inplace=True),            
#             nn.Conv2d(512,256,kernel_size=1,stride=1,padding=1//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(256,512,kernel_size=3,stride=1,padding=3//2),
#             nn.LeakyReLU(0.1,inplace=True),            
#             nn.Conv2d(512,256,kernel_size=1,stride=1,padding=1//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(256,512,kernel_size=3,stride=1,padding=3//2),
#             nn.LeakyReLU(0.1,inplace=True),      
#             nn.Conv2d(512,512,kernel_size=1,stride=1,padding=1//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=3//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.MaxPool2d(2,2),

#             # (1,1024,14,14) -> (1,1024,7,7)
#             nn.Conv2d(1024,512,kernel_size=1,stride=1,padding=1//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=3//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(1024,512,kernel_size=1,stride=1,padding=1//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=3//2),
#             nn.LeakyReLU(0.1,inplace=True),


#             # (1,1024,7,7) -> (1,1024,7,7)
#             nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=3//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(1024,1024,kernel_size=3,stride=2,padding=3//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=3//2),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=3//2),
#             nn.LeakyReLU(0.1,inplace=True),
#         )

#         #1番目のLinear　と　Local Layerの違い？
#         # (1,7*7*1024) -> (1,1470)
#         self.fc_layers=nn.Sequential(
#             nn.Linear(1024*7*7,4096),  #TODO: local layerの代わり　local layerって何？
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(4096,self.S*self.S*(self.B*5+self.num_classes)),
#             # nn.Sigmoid(),
#         )

#         self.sigmoid=nn.Sigmoid()
        


#     def forward(self,x):
#         out=self.model(x)
#         out=out.view(out.size(0),-1)
#         out=self.fc_layers(out)
#         #(channel,height,width)
#         out=out.view(-1,5*self.B+self.num_classes,self.S,self.S)
#         out=self.sigmoid(out)
#         #TODO: 全部にsigmoidかけてるけど、class_probabilityはsigmoidかけなくてもいいのでは？？？
#         boxes,class_probability_map=torch.split(out,[5*self.B,self.num_classes],dim=1)
#         return boxes,class_probability_map

#         # boxes,class_probability_map=torch.split(out,[self.B*5,self.num_classes],dim=1)
#         # print(boxes,class_probability_map)
#         # # boxes,class_probability_map=out
#         # boxes=torch.sigmoid(boxes)
#         # # class_probability_map=torch.nn.functional.softmax(class_probability_map,dim=1)


#         # # boxes: (batch_size,5,grid_size,grid_size)
#         # # 5のところは　x,y,height,width,confidence

#         # #NOTE: probability maxpはsoftmaxでクラス間の合計を1にしていない
#         # return boxes,class_probability_map
