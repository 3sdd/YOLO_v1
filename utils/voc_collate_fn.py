from torch.utils.data.dataloader import default_collate

def voc_collate_fn(batch_list):
    images = default_collate([batch_list[i][0] for i in range(len(batch_list))])
    
    annotations = []
    for i in range(len(batch_list)):
        annotations.append(batch_list[i][1]["annotation"])
    # for k in batch_list[0][1]['annotation']:
    #     annotations[k] = [batch_list[i][1]['annotation'][k] for i in range(len(batch_list))]

    for i in range(len(annotations)):
        obj=annotations[i]["object"]
        if not isinstance(obj,list):
            annotations[i]["object"]=[obj]

        # for obj in annotations[i]["object"]:
        #     # print(obj)
        #     # object_list = []

        #     if not isinstance(obj,list):
        #         # obj=[obj]
        #         # object_list.append(l)

        #         annotations[i]["object"]=[obj]

    # annotations['object'] = object_list
    return {'images':images,'annotations':annotations}
