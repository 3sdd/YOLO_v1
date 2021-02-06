import os
import torch
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import torchnet as tnt

from config import get_config
from models import TinyYOLOv1
import transforms as dt
from utils import voc_collate_fn,get_lr,save_checkpoint,save_result_images
from utils import class2index as voc_class2index
from utils import index2class as voc_index2class
from loss import YOLOLoss

def train():
    config=get_config()
    print("設定")
    print(config)

    #フォルダー作成
    os.makedirs(config.result_dir,exist_ok=True)
    os.makedirs(config.checkpoint_dir,exist_ok=True)
    result_image_dir=os.path.join(config.log_dir,"result_images")
    os.makedirs(result_image_dir,exist_ok=True)


    model=TinyYOLOv1(num_classes=config.num_classes,S=config.S,B=config.B)
    model=model.to(config.device)
    print("model")
    print(model)

    optimizer=optim.Adam(model.parameters(),lr=config.lr)
    criterion=YOLOLoss(noobject_scale=config.noobject_scale,coord_scale=config.coord_scale,\
                    image_size=config.image_size,S=config.S,B=config.B,class2index=voc_class2index)

    print("optimizer:",optimizer)


    transforms_train=dt.VocTransforms(dt.Compose([
        dt.ColorJitter(saturation=0.5),
        dt.ToTensor(),
        dt.RandomGrayscale(p=0.05),
        dt.Resize(config.image_size),
        dt.RandomScale([0.5,1.5],p=0.5),
        dt.RandomHorizontalFlip(p=0.5)
    ]))
    transforms_test=dt.VocTransforms(dt.Compose([
        dt.ToTensor(),
        dt.Resize(config.image_size),
    ]))
    transforms_log=dt.VocTransforms(dt.Compose([
        dt.ToTensor(),
        dt.Resize(config.image_size),
    ]))

    root="./VOCDetection"
    dataset_train=torchvision.datasets.VOCDetection(root=root,year="2007",image_set="trainval",download=True,transforms=transforms_train)
    # dataset_test=torchvision.datasets.VOCDetection(root=root,year="2007",image_set="test",download=True,transforms=transforms_test)
    print("dataset(train) size:",len(dataset_train))
    # print("dataset(test) size:",len(dataset_test))

    dataloader_train=torch.utils.data.DataLoader(dataset_train,batch_size=config.batch_size,shuffle=True,collate_fn=voc_collate_fn,num_workers=config.num_workers)
    # dataloader_test=torch.utils.data.DataLoader(dataset_test,batch_size=config.batch_size,shuffle=False,collate_fn=voc_collate_fn,num_workers=config.num_workers)

    writer=SummaryWriter(log_dir=os.path.join(config.log_dir,"runs"))
    # writer=SummaryWriter(log_dir=os.path.join(config.result_dir,"runs"))

    print("学習スタート")
    model.train()

    losses=tnt.meter.AverageValueMeter()
    losses_localization=tnt.meter.AverageValueMeter()
    losses_classification=tnt.meter.AverageValueMeter()
    losses_confident=tnt.meter.AverageValueMeter()

    total_iter=0
    for epoch in range(1,config.num_epochs+1):
        #lossリセット
        losses.reset()
        losses_localization.reset()
        losses_classification.reset()
        losses_confident.reset()
        
        last_lr=get_lr(optimizer)  
        with tqdm(total=len(dataloader_train),position=0,desc="[train]") as pbar:
            pbar.set_description("[Epoch  %d/%d]"%(epoch,config.num_epochs))
            for i,d in enumerate(dataloader_train):

                images,annotations=d["images"],d["annotations"]
                total_iter+=images.size(0)
                images=images.to(config.device)

                boxes,p_maps=model(images)

                optimizer.zero_grad()    
                loss=criterion(boxes,p_maps,annotations)
                loss.backward()
                optimizer.step()

                #ロス保存
                batch_size=images.size(0)
                losses.add(loss.item(),batch_size)
                losses_localization.add(criterion.localization_loss,batch_size)
                losses_classification.add(criterion.classification_loss,batch_size)
                losses_confident.add(criterion.confident_loss,batch_size)

                if writer is not None:
                    writer.add_scalar("Loss/train",losses.mean,total_iter)
                    writer.add_scalar("ClassificationLoss/train",losses_classification.mean,total_iter)
                    writer.add_scalar("LocalizationLoss/train",losses_localization.mean,total_iter)
                    writer.add_scalar("ConfidentLoss/train",losses_confident.mean,total_iter)    

                pbar.set_postfix({"lr":last_lr,"loss":losses.mean,"loss(localization)":losses_localization.mean,\
                                "loss(classification)":losses_classification.mean,"loss(confident)":losses_confident.mean})
                pbar.update()

        
        # if config.use_lr_scheduler:
        #     lr_scheduler.step()


        # validate(model,dataloader["val"],config,epoch,writer)
        save_checkpoint(os.path.join(config.checkpoint_dir,"checkpoint.pth"),model,optimizer,config,epoch)

        threshold=0.5
        save_result_images(result_image_dir,model,dataloader_train.dataset,\
            [0,1,2,3],voc_collate_fn,transforms_log,config,threshold,epoch,voc_index2class)


    print("学習終わり")
    torch.save(model.state_dict(),os.path.join(config.result_dir,"model_last_state_dict.pth"))
    
    writer.close()


if __name__=="__main__":
    train()




