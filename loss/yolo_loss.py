import torch
import torchvision
import torch.nn.functional as F
import math

from .util import get_center, to_relative_center_xy,prediction2truebox


class YOLOLoss():
    def __init__(self,noobject_scale,coord_scale,image_size,S,B,class2index):
        self.noobject_scale=noobject_scale # lambda_noobj
        self.coord_scale=coord_scale      # lambda_cord 

        self.image_size=image_size
        self.S=S
        self.B=B

        self.class2index=class2index

        self.cell_size=image_size/S

        #ロス保存用
        #バッチ数分の各項の合計値が入っている
        self.item1=0 
        self.item2=0
        self.item3=0
        self.item4=0
        self.item5=0
        

    def __call__(self,boxes,p_maps,annotations):
        """
        返り値はlossで　バッチ１つあたりのloss。注意:バッチ分のlossの合計値をbatch sizeで割ってある。
        batchで処理
            boxes: tensor  (batch_size,10,7,7)
            p_maps: tensor (batch_size,20,7,7),
            annotations:list (batch_size,)
        """
        self.clear_saved_loss_items()
        batch_size=boxes.size(0)
        device=boxes.device
        num_classes=p_maps.size(1)

        loss=0

        #label 
        gt_label=torch.full((batch_size,self.S,self.S),-1,device=device,dtype=torch.int) #-1 umeru

        #target_box_tensorは最終的に(batch_size,5*B,S,S)になる。　値をセットした後に繰り返しで増やすので最初は(batch_size,5,S,S)
        #で値を設定していく
        target_box_tensor=torch.zeros((batch_size,5,self.S,self.S),device=device,dtype=torch.float)
        target_class_tensor=torch.zeros((batch_size,num_classes,self.S,self.S),device=device,dtype=torch.float)

        gt_objects_xyxy=[[] for i in range(batch_size)] #[[(cell_x,cell_y,xyxy_tensor)] , [..]]
        predicted_objects_xyxy=[[] for i in range(batch_size)] #gt_objects_xywhcに対応する予測したbounding boxの(xmin,ymin,xmax,ymax)のもの
        mask_responsible=torch.zeros((batch_size,1,self.S,self.S),device=device,dtype=torch.float)


        #複数物体が１つのセルに入った時の処理、2回目以降に出てきた物体は無視する
        #(x,y)を保持しておき、2回目の(x,y)の時は処理を飛ばす。 for obj in objects:の中で行う
        for batch_idx in range(batch_size):
            objects=annotations[batch_idx]["transformed"]["object"]#target_transformで作成した部分
            done=set()

            for obj in objects:
                label=self.class2index(obj["name"]) #TODO:このクラス名からindexへの変換はtransformでするべきでは？
                bndbox=obj["bndbox"] 
                #物体の座標(abs). resizeされているのでfloatになっている

                xmin,ymin=bndbox["xmin"],bndbox["ymin"]
                xmax,ymax=bndbox["xmax"],bndbox["ymax"]
                cell_x,cell_y= self.get_cell_xy_index(xmin,ymin,xmax,ymax)
                # print(done)
                if (cell_x,cell_y) in done:
                    # print("already exists",(cell_x,cell_y))
                    continue
                gt_label[batch_idx,cell_y,cell_x]=label
                target_class_tensor[batch_idx,label,cell_y,cell_x]=1 #target class 

                gt_list_xywh=self.get_target_xywh(xmin,ymin,xmax,ymax) 
                target_box_tensor[batch_idx,:,cell_y,cell_x]=torch.tensor([*gt_list_xywh,1],device=device) #TODO:ここのconfidenceが1でいいのか #いらない？
                gt_box_xywh=torch.tensor(gt_list_xywh,device=device,dtype=torch.float)

                gt_box=torch.tensor([xmin,ymin,xmax,ymax],device=device)
                gt_objects_xyxy[batch_idx].append((cell_x,cell_y,gt_box))
                # print(boxes[batch_idx,:,cell_y,cell_x])
                # print(boxes[batch_idx,:,cell_y,cell_x].size()) #10
                # print(boxes[batch_idx,:,cell_y,cell_x].view(self.B,-1))
                # print(torch.split(boxes[batch_idx,:,cell_y,cell_x],[5 for x in range(self.B)],dim=0))
                # print(torch.stack(torch.split(boxes[batch_idx,:,cell_y,cell_x],[5 for x in range(self.B)],dim=0)))

                #予測したB個のbounding boxを (B,5)の形にする
                predicted_boxes=torch.stack(torch.split(boxes[batch_idx,:,cell_y,cell_x],[5 for x in range(self.B)],dim=0))
                predicted_boxes_xywh=predicted_boxes[:,:4]

                #(x,y,w,h)を(xmin,ymin,xmax,ymax)の形に直す
                tmp_array=[]
                for b in range(self.B):
                    box=predicted_boxes[b]
                    box=prediction2truebox(box,cell_x,cell_y,self.image_size,self.cell_size)
                    tmp_array.append(box)
                    
                #size=(B,4)
                predicted_boxes_xyxy=torch.stack(tmp_array) 
                predicted_objects_xyxy[batch_idx].append(predicted_boxes_xyxy)

                #ground truth物体1つと予測したbounding boxB個のiou計算
                #cell_x,cell_yにおけるresponsibleな予測を見つける
                
                gt_box=gt_box.view(1,-1)
                iou=torchvision.ops.box_iou(gt_box,predicted_boxes_xyxy)

                #dimはいらない？
                idx=torch.argmax(iou) #iouが大きい方のindex 
                #ロスの項1,2,3を計算
                target_box=predicted_boxes[idx]
                gt_box=gt_box_xywh

                #TODO:gt_boxを相対に

                #target_boxとgt_boxを使って　ロスを計算する
                term1=self.coord_scale*(torch.pow(target_box[0]-gt_box[0],2)+\
                                        torch.pow(target_box[1]-gt_box[1],2))
                term2=self.coord_scale*(torch.pow(target_box[2].sqrt()-gt_box[2].sqrt(),2)+\
                                        torch.pow(target_box[3].sqrt()-gt_box[3].sqrt(),2))

                term3=torch.pow(target_box[4]-1,2) #TODO:ここの1は本当はiou? iou=iou[0,idx]


                self.item1+=term1.item()
                self.item2+=term2.item()
                self.item3+=term3.item()
                
                loss+=(term1+term2+term3)
                done.add((cell_x,cell_y))

        #     #TODO: x,y,w,h,c から　xmin,ymin,xmax,ymaxへと変換する必要がある
        #     iou_matrix=torchvision.ops.box_iou(gt_boxes,predicted_boxes)
        #     print("iou")
        #     print(iou_matrix)
        #     print(iou_matrix.size())
            
        mask_object=gt_label>=0
        mask_no_object=gt_label==-1

        #responsibleなものを見つけるためにiouが一番高いものを求める
        # box_iou=torchvision.ops.box_iou(prediction_boxes,gt_boxes_coord)

        idx=[5*x+4 for x in range(self.B)] #confidenceのindexを集めたリスト
        # print(idx)
        tmp_bs=2

        # print(mask_object)
        # print(mask_object.size())
        tmp_mask=torch.tensor([[True,True],[False,False]]).unsqueeze(0)
        # print(tmp_mask)
        # print(tmp_mask.size())
        # print((boxes[:tmp_bs,idx,:2,:2]-0).pow(2)*tmp_mask)

        # print(boxes[:,idx,:,:].size()) #(batch_size,B,S,S)

        #no object confidence loss
        confidence_tensor=boxes[:,idx,:,:] #(batch_size,B,S,S)
        #TODO:maskかけたが、あってるかわからない。たぶんあってる気がする
        # print(mask_no_object.unsqueeze(1).size())
        # print((self.noobject_scale*(confidence_tensor-0).pow(2)).size())

        # print(mask_no_object.unsqueeze(1))
        # print(self.noobject_scale*(confidence_tensor-0).pow(2))
        # print(self.noobject_scale*(confidence_tensor-0).pow(2)*mask_no_object.unsqueeze(1))
        tmp=self.noobject_scale*(confidence_tensor-0).pow(2)*mask_no_object.unsqueeze(1) #broaadcast?
        term4=tmp.sum()
        # print(term4)

        loss+=term4
        # print(boxes[0,:,0,0])

        #boxes (bs,5*B,7,7)
        # print(boxes[:,[0,1],:,:])
        # c1=torch.pow(boxes[:,5,:,:]-0,2)
        # c2=torch.pow(boxes[:,5,:,:]-0,2)


        # class loss
        #TODO: オブジェクトが存在している場所のみロス計算するのでマスクかける
        # p_maps,target_class_tensor　どちらもsize=(batch_size,num_classes,S,S)

        p_maps_only_object_exists=p_maps*mask_object.unsqueeze(1)

        term5= F.mse_loss(p_maps_only_object_exists,target_class_tensor,reduction="sum")
        loss+=term5

        self.item4+=term4.item()
        self.item5+=term5.item()


        self.item1=self.item1/batch_size
        self.item2=self.item2/batch_size
        self.item3=self.item3/batch_size
        self.item4=self.item4/batch_size
        self.item5=self.item5/batch_size

        return loss/batch_size

    def clear_saved_loss_items(self):
        self.item1=0
        self.item2=0
        self.item3=0
        self.item4=0
        self.item5=0

    def localization_loss(self):
        return self.item1+self.item2
    def confident_loss(self):
        return self.item3+self.item4
    def classification_loss(self):
        return self.item5

    def get_cell_xy_index(self,gt_xmin,gt_ymin,gt_xmax,gt_ymax):
        """
        cellの(x,y)を取得する　
        引数はすべて絶対座標
        """
        #絶対座標
        center_x,center_y=get_center(gt_xmin,gt_ymin,gt_xmax,gt_ymax)

        #相対座標へ変換
        rel_center_x,rel_center_y=center_x/self.image_size,center_y/self.image_size

        #gridのどの位置にあるか調べる
        cell_size=1/self.S
        cell_idx_x=math.floor(rel_center_x/cell_size)
        cell_idx_y=math.floor(rel_center_y/cell_size)
        #rel_center_*/cell_sizeはrel_center_*==1の時　indexが最大のもの超えるので、一つしたのに下げる
        cell_idx_x= self.S if cell_idx_x>=self.S else cell_idx_x
        cell_idx_y= self.S if cell_idx_y>=self.S else cell_idx_y

        return cell_idx_x,cell_idx_y
    def get_target_xywh(self,xmin,ymin,xmax,ymax):
        x,y=to_relative_center_xy(xmin,ymin,xmax,ymax,self.cell_size)
        w=(xmax-xmin)/self.image_size
        h=(ymax-ymin)/self.image_size
        return [x,y,w,h]

# criterion=YOLOLoss(noobject_scale=config.noobject_scale,coord_scale=config.coord_scale,\
#                 image_size=config.image_size,S=config.S,B=config.B)


# #bboxes: size(batch_size,10,7,7)
# #p_maps: size(batch_size,20,7,7)
# bboxes,p_maps=model(imgs.to(config.device))
# loss=criterion(imgs,bboxes,p_maps,anno)
# print("loss:",loss)
# loss.backward()