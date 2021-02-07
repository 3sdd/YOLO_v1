
import torch
import torchvision
from PIL import ImageDraw
import os

from .drawing import draw_boxes_on_image

def get_lr(optimizer):
    """optimizerのlrを取得する
    """

    lr=[]
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return lr

def save_checkpoint(path:str,model,optimizer,config,epoch:int):
    """checkpointを保存する

    Args:
        path (str): 保存先のパス
        model ([type]): [description]
        optimizer ([type]): 保存するOptimizer
        config ([type]): [description]
        epoch (int): 何epoch目か
    """
    torch.save({
        "epoch":epoch,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "config":config
    },path)


def save_result_images(result_dir,model,dataset,dataset_indices,collate_fn,transforms,config,threshold,epoch,index2class):
    is_training=model.training
    model.eval()
    tmp=dataset.transforms
    dataset.transforms=transforms
    with torch.no_grad():
        for index in dataset_indices:
            imgs,annotations=collate_fn([dataset[index]])
            img,annotation=imgs[0],annotations[0]
            img=img.to(config.device)
            objects=annotation["transformed"]["object"]
            out,p_map=model(img.unsqueeze(0))

            img_pil=torchvision.transforms.functional.to_pil_image(img)
            
            #検出結果画像を保存
            bounding_boxes=yoloresult2bboxes(out,p_map,B=config.B,S=config.S,threshold=threshold)
            image_predicted=img_pil.copy()
            image_predicted=draw_boxes_on_image([image_predicted],bounding_boxes,config.S,index2class=index2class)[0]
            path=os.path.join(result_dir,f"image_predicted_{index}_epoch{epoch}.jpg")
            image_predicted.save(path)
    dataset.transforms=tmp
    
    if is_training:
        model.train()

#TODO:返り値を辞書型からlist型にする
@torch.no_grad()
def yoloresult2bboxes(boxes,p_map,B,S,threshold=0.5):
    """YOLOの予測結果からbounding boxへ変える

    返り値は list型で、各要素は下の辞書型。boxには予測値のx,y,w,h,cがlabelには
    いまはまだNoneしか入っていない。
    {
        "box":[x,y,w,h,c],
        "label":None,
    }
    """
    boxes=boxes.cpu()
    ret_boxes_list=[]

    for batch_idx in range(boxes.size(0)):
        ret_boxes={}
        for b in range(B):
            box=boxes[batch_idx][5*b:5*b+5]
            box_confidence=box[4,:,:]

            over_threshold=box_confidence>threshold
            
            # print(over_threshold)
            for y in range(S):
                for x in range(S):
                    if not over_threshold[y,x]:
                        continue
                    probability=p_map[batch_idx][:,y,x]

                    #TODO:probabilityをsum()しても1にはならないのだが。。
                    # 1になるようにした方がいいのかな？
                    label=torch.argmax(probability).item()
                    max_value=probability[label].item()

                    box_item=box[:,y,x]
                    if (x,y) in ret_boxes:
                        # ret_boxes[(x,y)].append(box_item)
                        ret_boxes[(x,y)].append({
                            "box":box_item,
                            "label":label,
                            "probability":max_value
                        })
                    else:
                        ret_boxes[(x,y)]=[{
                            "box":box_item,
                            "label":label,
                            "probability":max_value
                        }]

        ret_boxes_list.append(ret_boxes)

    return ret_boxes_list


def clamp(x:float, min:float, max:float)->float:
    """指定された値xをmin<=x<=maxの範囲に収まるようにx<minはminにx>maxはmaxにする

    Args:
        x (float): 元の値
        min (float): 最小値。xがこの値より小さい時はx=minとなる値を返す
        max (float): 最大値。xがこの値より大きい時はx=maxとなる値を返す

    Returns:
        float: min<=x<=maxの範囲に収まるようにした値
    """
    if x<min:
        return min
    elif x>max:
        return max
    else:
        return x
