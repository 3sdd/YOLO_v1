
from PIL import ImageDraw, ImageFont
import torch
# from imgaug.augmentables.bbs import BoundingBox,BoundingBoxesOnImage
import numpy as np
from PIL import Image
import torchvision
import os

font_path=os.path.join(os.path.dirname(__file__),"../fonts/ttf/KleeOne-Regular.ttf")
font=ImageFont.truetype(font_path,20)

# def draw_gt_bboxes(image,annotation_object,copy=True,color=(0,255,0),size=1):
#     if copy:
#         image=image.copy()
#     # bb_list=[]
#     # print(annotation_object)
#     image=np.asarray(image)
#     for obj in annotation_object:
#         # print(obj)
#         bndbox=obj["bndbox"]
#         xmin,ymin=bndbox["xmin"],bndbox["ymin"]
#         xmax,ymax=bndbox["xmax"],bndbox["ymax"]
#         bb=BoundingBox(xmin,ymin,xmax,ymax,label=obj["name"])
#         image=bb.draw_on_image(image,color=color,size=size)
#     image=Image.fromarray(image)
#     return image

# #TODO: cell_size=8 などを引数にいれてグリッド表示するようにする　cell_size=None:表示しない
# def draw_bounding_boxes(image,bounding_box_list,size=2,color=[255,0,0]):
#     """画像にYOLOの検出結果を書き込む
#         image (Tensor|PIL.Image): tensor画像 or pilの画像 
#         bounding_box_list (list): bounding boxの情報　[{'coordinate':[xmin,ymin,xmax,ymax], 'label':label},....] 
#         [{'coordinate': [10,10,20,20],'label':'car'}]

#     """
#     #画像を tensor-> pil image
#     # if isinstance(image):
#     #     pass
#     if isinstance(image,torch.Tensor):
#         image=torchvision.transforms.functional.to_pil_image(image)
#     # else:
#     #     raise Exception("Invalid Image Type (must  be pil or tensor)",type(image))
#     image=np.array(image)

#     for box in bounding_box_list:
#         xmin,ymin,xmax,ymax=box["coordinate"]
#         label=box["label"]
#         bb=BoundingBox(xmin,ymin,xmax,ymax,label=label)
#         image=bb.draw_on_image(image,color=color,size=size)

#     return Image.fromarray(image)


def draw_grid_on_image(image,S,width=0,fill="black",copy=True):
    """画像にグリッドを書き込む

    Args:
        image (PIL.Image): 元となるPIL画像
        S (int): グリッドの分割数
        width (int, optional): [description]. Defaults to 0.
        fill (str, optional): [description]. Defaults to "black".
        copy (bool, optional): 画像をコピーして書き込むとかどうか. Defaults to True.

    Returns:
        [PIL.Image]: グリッドを書き込んだ画像
    """
    image_size=image.size[0]
    grid_size=image_size/S

    if copy:
        image=image.copy()
    draw=ImageDraw.Draw(image)
    #grid cell 描画
    #枠は右端と下端が描画されなかったので、range(S+1)ではなくrange(1,S)にした
    #横軸
    for xline_idx in range(1,S):
        draw.line([(0,grid_size*xline_idx),(image_size,grid_size*xline_idx)],fill=fill,width=width)
    #縦軸
    for yline_idx in range(1,S):
        draw.line([(grid_size*yline_idx,0),(grid_size*yline_idx,image_size)],fill=fill,width=width)
    
    return image

#TODO:ラベルも表示もっと見やすくしたい
def draw_boxes_on_image(image,bounding_boxes,S=7,draw_grid=True,color="red",index2class=None):
    text_fill=(255,255,255,255)
    # bounding_boxes [dict] : { "(x,y)":{"box":[x,y,w,h,c],"label":None} , "(x,y)": ....}
    
    #pil画像に変換する
    if isinstance(image,torch.Tensor):
        img_l=[]
        for batch_idx in range(image.size(0)):
            img_l.append(torchvision.transforms.functional.to_pil_image(image[batch_idx]))
        image=img_l

    # print(type(image[batch_idx]))
    image_size=image[0].size[0] #PIL image size
    grid_size=image_size/S

    for batch_idx in range(len(image)):
        img=image[batch_idx]
        img=img.copy()

        if draw_grid:
            img=draw_grid_on_image(img,S)
        d=ImageDraw.Draw(img)

        # print(final_prediction[batch_idx])
        for grid_y in range(S):
            for grid_x in range(S):
                if  (grid_x,grid_y) not in bounding_boxes[batch_idx].keys():
                    continue

                f_bbs=bounding_boxes[batch_idx][(grid_x,grid_y)]
                for bbox in f_bbs:
                    #bbox
                    bb=bbox["box"]
                    
                    x,y,width,height,confidence=bb.tolist()# size :5 
                    center_x,center_y=grid_x*grid_size+grid_size*x,grid_y*grid_size+y*grid_size

                    width,height=width*image_size,height*image_size
                    x1,y1=center_x-width/2,center_y-height/2
                    x2,y2=center_x+width/2,center_y+height/2
                    # print([x1,y1,x2,y2])
                    d.rectangle([x1,y1,x2,y2],outline=color)   

                    #ラベル 
                    label=bbox["label"]
                    if index2class is not None:
                        label=index2class(label)
                    d.text((x1,y1),label,fill=text_fill,font=font)

        image[batch_idx]=img

    return image
