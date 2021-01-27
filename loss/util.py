import torch

def get_center(xmin,ymin,xmax,ymax):
    return (xmin+xmax)/2,(ymin+ymax)/2

def to_relative_center_xy(xmin,ymin,xmax,ymax,cell_size):
    """boxの座標(左上、右下の点)からboxの中心座標(cellに相対的)に変換する
    """
    abs_center_x,abs_center_y=get_center(xmin,ymin,xmax,ymax)
            #gtはabsなので　中心の座標のグリッドセルに対して相対の座標x_hat,y_hatにする
            
    x_rel_to_cell=(abs_center_x - abs_center_x//cell_size*cell_size)/cell_size
    y_rel_to_cell=(abs_center_y-abs_center_y//cell_size*cell_size)/cell_size

    return x_rel_to_cell,y_rel_to_cell



def prediction2truebox(xywh,cell_x,cell_y,image_size,cell_size):
    """
    returns tensor size(4)    xmin,ymin,xmax,ymax


    xywh: 予測したxywh。xyはセルに相対で[0,1]、whは画像サイズに相対で[0,1]。xywhはlistでサイズは4
    cell_x,cell_y : 画像を分割したセルのインデックス (x,y)。0スタート
    image_size: 画像のサイズ
    S:yoloのパラメータS
    """
    x,y=xywh[:2]
    w,h=xywh[2:4]

    center_x,center_y=cell_x*cell_size+cell_size*x,cell_y*cell_size+y*cell_size
    width,height=w*image_size,h*image_size
    half_width,half_height=width/2,height/2

    xmin,ymin=center_x-half_width,center_y-half_height
    xmax,ymax=center_x+half_width,center_y+half_height

    xyxy=[xmin,ymin,xmax,ymax]
    return torch.stack(xyxy)
