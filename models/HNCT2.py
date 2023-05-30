import torch
from torch import nn

from .common.modules import conv3x3, SwinModule,PatchMerging2
from .common.utils import up_sample
from .base_model import Base_model
from .builder import MODELS
from . import block as B
import torch.nn.functional as F

class CrossSwinTransformer(nn.Module):
    def __init__(self, cfg, logger, n_feats=64, n_heads=4, head_dim=16, win_size=4,
                 n_blocks=3, cross_module=['pan', 'ms'], cat_feat=['pan', 'ms'], sa_fusion=False,
                 in_nc = 5, nf = 64, num_modules=4):
        super().__init__()
        self.cfg = cfg
        self.n_blocks = n_blocks
        self.cross_module = cross_module
        self.cat_feat = cat_feat
        self.sa_fusion = sa_fusion

        self.pan_conv_first = nn.Conv2d(1,n_feats,3,1,1)
        self.pan_merge = PatchMerging2(n_feats,n_feats,4)
        self.ms_conv_first = nn.Conv2d(cfg.ms_chans,n_feats,3,1,1)
        self.conv_last = nn.Sequential(nn.Conv2d(n_feats, n_feats // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(n_feats // 4, n_feats // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(n_feats // 4, n_feats, 3, 1, 1))
        pan_encoder = [
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
                       #1,64,2,2,4,16,4,True,False
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
        ]
        ms_encoder = [
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),          
        ]

        
        self.HR_tail = nn.Sequential(
            conv3x3(n_feats, n_feats * 4),
            nn.PixelShuffle(2), nn.ReLU(True), conv3x3(n_feats, n_feats * 4),
            nn.PixelShuffle(2), nn.ReLU(True), conv3x3(n_feats, n_feats),
            nn.ReLU(True), conv3x3(n_feats, cfg.ms_chans))

        self.HR_tail2 = nn.Sequential(
            conv3x3(n_feats, n_feats * 4),
            nn.ReLU(True), conv3x3(n_feats * 4, n_feats * 4),
            nn.ReLU(True), conv3x3(n_feats * 4, n_feats),
            nn.ReLU(True), conv3x3(n_feats, cfg.ms_chans))
        
        self.pan_encoder = nn.Sequential(*pan_encoder)
        self.ms_encoder = nn.Sequential(*ms_encoder)

        #-----------------------以下为HNCT代码块部分-----------------------
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        self.B1 = B.HBCT(in_channels=nf)
        self.B2 = B.HBCT(in_channels=nf)
        self.B3 = B.HBCT(in_channels=nf)
        self.B4 = B.HBCT(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        upsample_block = B.pixelshuffle_block


    def forward(self, pan, ms, ms_u):

        lr_pan = F.interpolate(pan, scale_factor=1/2, mode='bicubic')   #PAN插值到1/2大小lr_pan
        ms_2 = up_sample(ms, r = 2)                                     #MS上采样到2倍大小ms_2
        x = torch.cat([lr_pan,ms_2],1)                                  #拼接起来 batch_size, 5, 64, 64
        out_fea = self.fea_conv(x)                                      #输入5通道，输出64通道特征，3*3卷积核
        out_B1 = self.B1(out_fea)                                       #第一个HBCT模块，输出64通道特征
        out_B2 = self.B2(out_B1)                                        #第二个HBCT模块，输出64通道特征
        out_B3 = self.B3(out_B2)                                        #第三个HBCT模块，输出64通道特征
        out_B4 = self.B4(out_B3)                                        #第四个HBCT模块，输出64通道特征
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))  #将四个HBCT模块concat起来，卷积，送入密集特征融合模块
                                                                            #输入通道为64*4，输出64
        out_lr = self.LR_conv(out_B) + out_fea  #进行密集特征融合，输入64输出64
        #output = self.upsampler(out_lr)     #送入上采样模块
        output = self.HR_tail2(out_lr)           #注入超分细节
        hrms2 = output + ms_2

        ms_4 = up_sample(hrms2, r = 2)
        x = torch.cat([pan,ms_4],1)  #batch_size, 5, 128, 128
        out_fea = self.fea_conv(x)   #输入5通道，输出64通道特征，3*3卷积核
        out_B1 = self.B1(out_fea)           #第一个HBCT模块，输出64通道特征
        out_B2 = self.B2(out_B1)            #第二个HBCT模块，输出64通道特征
        out_B3 = self.B3(out_B2)            #第三个HBCT模块，输出64通道特征
        out_B4 = self.B4(out_B3)            #第四个HBCT模块，输出64通道特征
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))  #将四个HBCT模块concat起来，卷积，送入密集特征融合模块
                                                                            #输入通道为64*4，输出64
        out_lr = self.LR_conv(out_B) + out_fea  #进行密集特征融合，输入64输出64
        #output = self.upsampler(out_lr)     #送入上采样模块
        output = self.HR_tail2(out_lr)           #注入超分细节
        output = output + ms_4
        #output = out_lr
        '''
        pan_feat = self.pan_conv_first(pan)
        pan_feat = self.pan_encoder(pan_feat)
        ms_feat = self.ms_conv_first(ms)
        ms_feat = self.ms_encoder(ms_feat) + ms_feat

        cat_list = []
        if 'pan' in self.cat_feat:
            cat_list.append(pan_feat)
        if 'ms' in self.cat_feat:
            cat_list.append(ms_feat)

        output = self.HR_tail(torch.cat(cat_list, dim=1))

        if self.cfg.norm_input:
            output = torch.clamp(output, 0, 1)
        else:
            output = torch.clamp(output, 0, 2 ** self.cfg.bit_depth - .5)
        '''

        return output


@MODELS.register_module()
class HNCT2(Base_model):
    def __init__(self, cfg, logger, train_data_loader, test_data_loader1):
        super().__init__(cfg, logger, train_data_loader, test_data_loader1)

        model_cfg = cfg.get('model_cfg', dict())
        G_cfg = model_cfg.get('core_module', dict())

        self.add_module('core_module', CrossSwinTransformer(cfg=cfg, logger=logger, **G_cfg))

    def get_model_output(self, input_batch):
        input_pan = input_batch['input_pan']
        input_lr = input_batch['input_lr']
        input_lr_u = up_sample(input_lr)
        output = self.module_dict['core_module'](input_pan, input_lr,input_lr_u)
        return output

    def train_iter(self, iter_id, input_batch, log_freq=10):
        G = self.module_dict['core_module']
        G_optim = self.optim_dict['core_module']
        input_pan_l = input_batch['input_pan_l']
        input_pan = input_batch['input_pan']
        input_lr = input_batch['input_lr']
        input_lr_u = up_sample(input_lr)
        output = G(input_pan, input_lr, input_lr_u)

        loss_g = 0
        loss_res = dict()
        loss_cfg = self.cfg.get('loss_cfg', {})
        if 'QNR_loss' in self.loss_module:
            QNR_loss = self.loss_module['QNR_loss'](pan = input_pan, ms = input_lr, pan_l = input_pan_l, out = output)
            loss_g = loss_g + QNR_loss * loss_cfg['QNR_loss'].w
            loss_res['QNR_loss'] = QNR_loss.item()
        '''
        if 'rec_loss' in self.loss_module:
            target = input_batch['target']
            rec_loss = self.loss_module['rec_loss'](
                out=output, gt=target
            )
            loss_g = loss_g + rec_loss * loss_cfg['rec_loss'].w
            loss_res['rec_loss'] = rec_loss.item()
        '''
        loss_res['full_loss'] = loss_g.item()

        G_optim.zero_grad()
        loss_g.backward()
        G_optim.step()

        self.print_train_log(iter_id, loss_res, log_freq)
