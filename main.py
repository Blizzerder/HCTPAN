import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import argparse
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import mmcv
from mmcv import Config
from mmcv.utils import get_logger
from logging import Logger
import traceback

from datasets.builder import build_dataset
from models.builder import build_model




def parse_args():
    parser = argparse.ArgumentParser(description='pan-sharpening implementation')
    parser.add_argument('-c', '--config', required=True,help='config file path')#设置config文件
    return parser.parse_args()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(cfg, logger):
    # type: (mmcv.Config, Logger) -> None
    # Setting Random Seed生成随机种子 此处不使用
    if 'seed' in cfg:               
        logger.info('===> Setting Random Seed')
        set_random_seed(cfg.seed, True)

    # Loading Datasets加载数据集
    logger.info('===> Loading Datasets')            #填写日志'===> Loading Datasets'
    if 'train_set_cfg' in cfg:                      #如果cfg中有'train_set_cfg'（训练集的参数设置）
        train_set_cfg = cfg.train_set_cfg.copy()    #train_set_cfg为cfg中设置的训练集的内容的复制
        train_set_cfg['dataset'] = build_dataset(cfg.train_set_cfg['dataset'])  #函数作用（有定义）：创建一个train数据集
        train_data_loader = DataLoader(**train_set_cfg)                         #加载数据
    else:
        train_data_loader = None

    test_set1_cfg = cfg.test_set1_cfg.copy()        #test_set1_cfg为cfg中设置的测试集set1
    test_set1_cfg['dataset'] = build_dataset(cfg.test_set1_cfg['dataset'])      #设置test的dataset
    test_data_loader1 = DataLoader(**test_set1_cfg)                             #加载数据

    # Building Model建立模型
    logger.info('===> Building Model')  #在日志中记录'===> Building Model'
    runner = build_model(cfg.model_type, cfg, logger, train_data_loader, test_data_loader1)  #建立模型
    #cfg.model_type："UCGAN" cfg:配置 logger：日志 train_data_loader：训练数据 test_data_loader0：测试数据0 test_data_loader1：测试数据1

    # Setting GPU设置GPU
    if 'cuda' in cfg and cfg.cuda:  #cuda=True
        logger.info("===> Setting GPU")
        runner.set_cuda()           #设置cuda

    # Weight Initialization权重初始化(修改：必定初始化)
    #if 'checkpoint' not in cfg:     
    logger.info("===> Weight Initializing")
    runner.init()

    # Resume from a Checkpoint (Optionally)从检查点恢复（修改：删除）
    '''
    if 'checkpoint' in cfg:
        logger.info("===> Loading Checkpoint")
        runner.load_checkpoint(cfg.checkpoint)
    '''
    # Copy Weights from a Checkpoint (Optionally)从检查点复制权重（修改：删除）
    '''
    if 'pretrained' in cfg:
        logger.info("===> Loading Pretrained")
        runner.load_pretrained(cfg.pretrained)
    '''
    # Setting Optimizer设置优化器
    logger.info("===> Setting Optimizer")
    runner.set_optim()

    # Setting Scheduler for learning_rate Decay设置learning_rate衰减的调度器
    logger.info("===> Setting Scheduler")
    runner.set_sched()

    # Print Params Count打印参数计数
    logger.info("===> Params Count")
    runner.print_total_params()
    runner.print_total_trainable_params()

    if ('only_test' not in cfg) or (not cfg.only_test):     #如果不是只测试模式就训练+保存（修改：必须训练+保存）
    # Training训练
        logger.info("===> Training Start")
        runner.train()

    # Saving保存
    logger.info("===> Final Saving Weights")    
    runner.save(iter_id=cfg.max_iter)

    # Testing测试
    logger.info("===> Final Testing")
    #runner.test(iter_id=cfg.max_iter, save=True, ref=True)  # low-resolution testing低分辨率测试
    runner.test(iter_id=cfg.max_iter, save=True, ref=False)  # full-resolution testing全分辨率测试

    logger.info("===> Finish !!!")

if __name__ == '__main__':
    args = parse_args()                 #参数设置
    cfg = Config.fromfile(args.config)  #函数作用：设置配置文件 args.config：config文件的位置(configs/panformer.py)来自launch.json 
    mmcv.mkdir_or_exist(cfg.log_dir)    #函数作用：创建日志文件夹 cfg.log_dir：日志文件位置(logs/ucgan)
    logger = get_logger('mmFusion', cfg.log_file, cfg.log_level)  #函数作用：创建logger实例 'mmFusion'：日志名 cfg.log_file：文件名（ucgan_GF-2） cfg.log_level：日志等级（INFO）  
    logger.info(f'Config:\n{cfg.pretty_text}') #函数作用：填写日志 内容：Config的全部信息

    try:
        main(cfg, logger)               #函数作用：进入main函数 cfg：config信息 logger：日志信息
    except:
        logger.error(str(traceback.format_exc()))