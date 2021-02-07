import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from collections.abc import Sequence

from utils import clamp

def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))

#bounding boxの座標
# 横: [0,image_width] , 縦: [0,image_height]
#画像全体を囲むのが (0,0,image_width,image_height)


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

        self.removed_index_list=[]

    def __call__(self, img, target):
        self.removed_index_list=[]

        index_list=list(range(len(target)))
        # print("index_list")
        # print(index_list)

        for i,t in enumerate(self.transforms):
            img, target = t(img, target)

            if t.removed_index_list is None:
                continue

            #transformしたあとremoved_index_listに含まれているindexを
            #それぞれのtransformに入力されたindexに対してではなく
            #全体でのindexとして保存する
            # print("removed index")
            # print(t.removed_index_list)
            #sortedは必要。大きいindexから処理することで、削除したあとでも同じ次に削除するindexは同じ
            for ri in sorted(t.removed_index_list,reverse=True):
                index=index_list.pop(ri)
                self.removed_index_list.append(index)

        self.removed_index_list.sort()

        return img, target

    # def __repr__(self):
    #     format_string = self.__class__.__name__ + '('
    #     for t in self.transforms:
    #         format_string += '\n'
    #         format_string += '    {0}'.format(t)
    #     format_string += '\n)'
    #     return format_string

class DetectionTransforms(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.removed_index_list=[]


#TODO:transforms.Resizeと同じような感じにしたい
#注意: サイズは (size,size)になる。正方形以外はまだできない
class Resize(DetectionTransforms):
    def __init__(self,size,interpolation=2):
        super().__init__()
        self.image_size=size
        self.interpolation=interpolation

    def forward(self,image,target):
        """
        image (Tensor): 変換する画像。　size=(channel,height,width)
        target (list):  画像に含まれている矩形情報。[[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],..]
        """
        original_img_height=image.size(1)
        original_img_width=image.size(2)

        #画像のリサイズ
        new_image=transforms.functional.resize(image,(self.image_size,self.image_size),self.interpolation)
        
        
        #矩形情報のリサイズ
        new_target=[]
        for i in range(len(target)):
            xmin,ymin=target[i][0],target[i][1]
            xmax,ymax=target[i][2],target[i][3]
            
            xmin= self.image_size/ original_img_width *xmin
            ymin=self.image_size/original_img_height*ymin
            xmax=self.image_size/original_img_width*xmax
            ymax=self.image_size/original_img_height*ymax
            new_target.append([xmin,ymin,xmax,ymax])

        return new_image,new_target

class ToTensor():
    def __init__(self):
        self.removed_index_list=[]

    def __call__(self, pic,target):
        return  transforms.functional.to_tensor(pic),target


class ColorJitter(DetectionTransforms):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()

        self.color_jitter=transforms.ColorJitter(brightness,contrast,saturation,hue)


    def forward(self,image,target):
        image=self.color_jitter(image)
        return image,target


class RandomGrayscale(DetectionTransforms):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, img, target):
        """ランダムにグレースケール化する

        Args:
            img (Tensor): 元画像

        Returns:
            Tensor: ランダムにグレースケール化した画像
        """
        num_output_channels = F._get_image_num_channels(img)
        if torch.rand(1) < self.p:
            return F.rgb_to_grayscale(img, num_output_channels=num_output_channels),target
        return img, target

class RandomHorizontalFlip(DetectionTransforms):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, target):
        if torch.rand(1) < self.p:
            width, height = F._get_image_size(img)
            #矩形情報を反転する
            # new_target=[]
            for i in range(len(target)):
                xmin,xmax=target[i][0],target[i][2]
                #xmin
                target[i][0]=width-xmax 
                #xmax
                target[i][2]=width-xmin 
                # new_target.append([xmin,ymin,xmax,ymax])

            return F.hflip(img), target
        return img,target
# class Translation(torch.nn.Module):
#     def __init__(self,translate):
#         """
#         translate (tuple): len(translate)==2
#         """
#         super().__init__()
#         _check_sequence_input(translate, "translate", req_sizes=(2, ))
#         for t in translate:
#             if not (0.0 <= t <= 1.0):
#                 raise ValueError("translation values should be between 0 and 1")
#         self.translate = translate


#         self.resample=0
#         # self.fillcolor=0 #tensorでfillcolorはつかえない？

#     def forward(self,image,target):
#         print(image.size())
#         w=image.size(2)
#         h=image.size(1)
#         print(w)
#         print(h)

#         dx = float(translate[0] * w)
#         dy = float(translate[1] * h)
#         translations = (dx, dy)

#         ret=(0, translations, 1.0, (0.0,0.0)) #(angle, translations, scale, shear)
#         image=transforms.functional.affine(image, *ret, resample=self.resample)

#         return image,target

class Scale(DetectionTransforms):
    def __init__(self,scale):
        super().__init__()
        # _check_sequence_input(scale, "scale", req_sizes=(2, ))
        # for s in scale:
        #     if s <= 0:
        #         raise ValueError("scale values should be positive")
        self.scale = scale

        self.resample=0
        # self.fillcolor=0 #tensorでfillcolorはつかえない？


    def forward(self,image,target):
        self.removed_index_list=[]
        # print("SCALER")
        #[channel,height,width]
        img_w=image.size(2)
        img_h=image.size(1)

        ret=(0, (0.0,0.0), self.scale, (0.0,0.0)) #(angle, translations, scale, shear)
        image= transforms.functional.affine(image, *ret, resample=self.resample)
    
        #画像をスケールしたあとの左上の座標
        #>1 px,pyはマイナスになる。<1の時　プラス
        px=(img_w-img_w*self.scale)/2
        py=(img_h-img_h*self.scale)/2
        new_bounding_boxes=[]
        for i,obj in enumerate(target):
            xmin,ymin=obj[0],obj[1]
            xmax,ymax=obj[2],obj[3]
            w,h=xmax-xmin,ymax-ymin

            #スケール後の画像の左上の座標(px,py)から　座標を scaleしてvalue*scaleしたものを加えて
            #scale後の座標にする
            # xmin,ymin=

            xmin,ymin=px+xmin*self.scale,py+ymin*self.scale
            xmax,ymax=px+xmax*self.scale,py+ymax*self.scale

            #画像外に出てしまった座標は画像内に収まるように 0 またはimg_w,img_hの値にする
            # print([xmin,ymin,xmax,ymax])
            xmin,ymin=clamp(xmin,0,img_w),clamp(ymin,0,img_h)
            xmax,ymax,=clamp(xmax,0,img_w),clamp(ymax,0,img_h)


            #削除後は、scaleの外側のtransformsでどう処理するか？annotationから削除する必要があるが、Composeで連続して処理する時はどうすればいいのか？
            #画像からはみ出たbounding box(面積が0になっているはず)を削除する
            if (xmax-xmin)*(ymax-ymin)==0:
                self.removed_index_list.append(i)                
            else:
                new_bounding_boxes.append([xmin,ymin,xmax,ymax])
        return image,new_bounding_boxes

class RandomScale(DetectionTransforms):

    def __init__(self,scale_ranges,p=0.5):
        """
            pの確率でscaleする。scaleは　sale_ranges=(min,max)
        """
        super().__init__()

        # if scale is not None:
        # _check_sequence_input(scale, "scale", req_sizes=(2, ))
        # for s in scale:
        #     if s <= 0:
        #         raise ValueError("scale values should be positive")
        # self.scale = scale
        self.scale_ranges=scale_ranges
        self.p=p

        self.previous_scale=None
    
        self.transforms_sacle=Scale(1)


    def forward(self,image,target):
        self.removed_index_list=[]
        if torch.rand(1) < self.p:
            #scale_rangesからランダムにscaleを決める
            scale = float(torch.empty(1).uniform_(self.scale_ranges[0], self.scale_ranges[1]).item())
            self.previous_scale=scale
            #scaleを適用
            self.transforms_sacle.scale=scale
            #実行
            image,target=self.transforms_sacle(image,target)
            self.removed_index_list=self.transforms_sacle.removed_index_list

        return image,target


#scale (0.5) scale(1.5)でもとの画像、bounding boxの座標になるか？
