from torch.utils.data.dataloader import default_collate

def voc_collate_fn(batch_list):
    images = default_collate([batch_list[i][0] for i in range(len(batch_list))])
    
    annotations = [item[1]["annotation"] for item in batch_list]

    return {'images':images,'annotations':annotations}
