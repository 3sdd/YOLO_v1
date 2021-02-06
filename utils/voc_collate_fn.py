from torch.utils.data.dataloader import default_collate

def voc_collate_fn(batch_list):
    images = default_collate([batch_list[i][0] for i in range(len(batch_list))])
    
    annotations = []
    for i in range(len(batch_list)):
        annotations.append(batch_list[i][1]["annotation"])

    for i in range(len(annotations)):
        obj=annotations[i]["object"]
        if not isinstance(obj,list):
            annotations[i]["object"]=[obj]

    return {'images':images,'annotations':annotations}
