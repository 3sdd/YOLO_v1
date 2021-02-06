from torch.utils.data.dataloader import default_collate


voc_classes=[
    "person",
    "bird",
    "cat",
    "cow",
    "dog",
    "horse",
    "sheep",
    "aeroplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "motorbike",
    "train",
    "bottle",
    "chair",
    "diningtable",
    "pottedplant",
    "sofa",
    "tvmonitor",
]


def class2index(class_name:str):
    return voc_classes.index(class_name)

def index2class(index:int):
    return voc_classes[index]



def voc_collate_fn(batch_list):
    images = default_collate([batch_list[i][0] for i in range(len(batch_list))])
    
    annotations = [item[1]["annotation"] for item in batch_list]

    return {'images':images,'annotations':annotations}
