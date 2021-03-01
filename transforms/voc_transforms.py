import copy

#image_transform -> 
class VocTransforms():
    def __init__(self,transform):
        """
        voc detectionのtransforms
        transform : forward(image,target)をもつtransform。removed_index_listももつ。
        """
        self.transform=transform


    def __call__(self,img,target):
        anno=target["annotation"]

        dic={
            # "size":{"width":self.image_size,"height":self.image_size,"depth":3},
            "object":[]
        }

        original_img_width=int(anno["size"]["width"])
        original_img_height=int(anno["size"]["height"])

        #画像のみのtransformを適用
        # img=self.transform_image(img)

        if isinstance(anno["object"],list):
            objects=anno["object"]
        else:
            objects=[anno["object"]]

        bbox_objects=[]
        for obj in objects:
            # obj=obj.copy() # pythonのfor .. in .. で作った変数ってcopyされたやつ？一応copyしておく
            obj=copy.deepcopy(obj)  # コピー前に反映されないようにコピーしておく

            bndbox=obj["bndbox"]
            label=obj["name"]
            xmin,ymin=int(bndbox["xmin"]),int(bndbox["ymin"])
            xmax,ymax=int(bndbox["xmax"]),int(bndbox["ymax"])
            #TODO: これ調べる
            # pascal voc の(xmin,ymin)を左上の頂点が(0,0)となる座標にするために-1する
            xmin,ymin=xmin-1,ymin-1

            bbox_objects.append([xmin,ymin,xmax,ymax,label])
            dic["object"].append(obj)

        #bounding boxにも影響があるtransformを適用
        img,bbox_objects=self.transform(img,bbox_objects)
        removed_index_list=self.transform.removed_index_list


        list_objects=dic["object"]
        #transformで削除された物体を消す
        for i in sorted(removed_index_list,reverse=True):
            del list_objects[i]

        
        #TODO: 扱いやすい形式にしたい
        for i in range(len(list_objects)):
            xmin,ymin,xmax,ymax,label=bbox_objects[i]
            bndbox=dic["object"][i]["bndbox"]
            
            bndbox["xmin"]=xmin
            bndbox["ymin"]=ymin
            bndbox["xmax"]=xmax
            bndbox["ymax"]=ymax

            dic["object"][i]["bndbox"]=bndbox
        target["annotation"]["transformed"]=dic

        return img,target