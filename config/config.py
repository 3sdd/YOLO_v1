import torch
import pprint

class Config():
    def __repr__(self):
        return self.__class__.__name__ + pprint.pformat(self.__dict__)


def get_config():
    c=Config()
    c.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # c.device=torch.device("cpu")

    c.batch_size=2
    c.num_epochs=100
    c.num_workers=2
    c.image_size=448 #画像サイズ  (channel,height,width)=(3,image_size,image_size)

    c.num_classes=20  # C, クラス数

    c.S=7  # S: グリッドの分割数
    c.B=2  # B: 1つのcellで予測するbounding boxの個数

    #loss
    c.object_scale=1
    c.noobject_scale=0.5
    c.class_scale=1
    c.coord_scale=5
    
    # c.lr=1e-2
    c.lr=1e-3 
    # c.momentum=0.9
    # c.weight_decay=5e-5

    c.result_dir="./results"
    c.log_dir="./logs"
    c.checkpoint_dir="./checkpoint"

    c.resume=False

    # c.use_lr_scheduler=False


    return c