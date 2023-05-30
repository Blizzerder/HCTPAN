from mmcv.utils import Registry
from mmcv import Config
import torch.utils.data as data

# create a registry
DATASETS = Registry('dataset')  #Register是一种实例化工具 这里创建了一个全局注册器嘞DATASETS


# create a build function
def build_dataset(cfg: Config, *args, **kwargs) -> data.Dataset:    #cfg.train_set_cfg['dataset']
    cfg_ = cfg.copy()                   #复制训练集的config信息
    dataset_type = cfg_.pop('type')     #获取dataset类型信息：PSDataset顺序数据集
    if dataset_type not in DATASETS:
        raise KeyError(f'Unrecognized task type {dataset_type}')
    else:
        dataset_cls = DATASETS.get(dataset_type)    #函数作用：获取对应的key（可以理解为字典） dataset_type：config得到的数据集类型

    #！！！！！！转到ps_dataset.py文件
    dataset = dataset_cls(*args, **kwargs, **cfg_)  #函数作用：创建数据集 cfg_：config信息
    return dataset      #返回一个带有pan的图像id和文件路径的文件夹
