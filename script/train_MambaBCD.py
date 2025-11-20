import sys
sys.path.append('/mnt/workspace/MambaCD')

import argparse
import os
import time
import datetime

import numpy as np

from changedetection_c.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm
from changedetection_c.datasets.make_data_loader import ChangeDetectionDatset, make_data_loader
from changedetection_c.utils_func.metrics import Evaluator
from changedetection_c.models.MambaBCD import STMambaBCD
from typing import Any, BinaryIO, List, Optional, Tuple, Union
import matplotlib.pyplot as plt

import changedetection_c.utils_func.lovasz_loss as L
import changedetection_c.utils_func.dice_loss as D
torch.cuda.empty_cache()
# 设置内存管理配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """

# def make_numpy_grid(tensor_data, pad_value=0, padding=0):
#     tensor_data = tensor_data.detach()
#     vis = make_grid(tensor_data, pad_value=pad_value, padding=padding)
#     print("tensor_data:", tensor_data)
#     print("vis:", vis)
#     vis = np.array(vis.cpu()).transpose((1, 2, 0))
#     if vis.shape[2] == 1:
#         vis = np.stack([vis, vis, vis], axis=-1)
#     return vis

# def make_numpy_grid(tensor_data, pad_value=0, padding=0):
#     tensor_data = tensor_data.detach()
#     vis = np.array(tensor_data)
#     vis = np.transpose(vis, (1, 2, 0))
#     if vis.shape[2] == 1:
#         vis = np.stack([vis, vis, vis], axis=-1)
#     return vis

def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    tensor_data = tensor_data.detach().cpu()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis).transpose((1, 2, 0))
    # vis = np.array(tensor_data)
    # vis = np.transpose(vis, (1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    if vis.shape[2] == 2:
        channel = vis[:, :, 1:2]
        vis = np.concatenate([channel, channel, channel], axis=2)
    return vis

def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)

        self.evaluator = Evaluator(num_class=2)

        self.deep_model = STMambaBCD(
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM, 
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            ) 
        self.epoch = args.max_iters // args.batch_size
        self.deep_model = self.deep_model.cuda()
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,f'1120_nofre{self.epoch}')
        self.lr = args.learning_rate


        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)  # 创建这个目录

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
        
    
    def training(self):
        best_f1_score = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        for _ in tqdm(range(elem_num)):   # tqdm：进度条
            itera, data = train_enumerator.__next__()  # for index, fruit in enumerator(dataset):print(index, data) 所以itera代表索引值
            # itera_count = 0
            # for itera in train_enumerator:
            #     itera_count += 1
            # print("number of itera", itera_count)
            pre_change_imgs, post_change_imgs, labels, _ = data
            
            pre_change_imgs = pre_change_imgs.cuda().float()  # 转到GPU并转换成浮点型
            post_change_imgs = post_change_imgs.cuda()   #torch.Size([4, 3, 256, 256])
            labels = labels.cuda().long()    # 长整型    torch.Size([4, 256, 256])
            
            out_put1 = self.deep_model(pre_change_imgs, post_change_imgs)#torch.Size([4, 2, 256, 256])
            
            self.optim.zero_grad()

            out_put = F.softmax(out_put1, dim=1)  # torch.Size([4, 2, 256, 256])
            out_put2 = out_put.argmax(dim=1)  #torch.Size([4, 256, 256])
            out_put2 = out_put2.unsqueeze(1)
            # 损失
            ce_loss_1 = F.cross_entropy(out_put1, labels, ignore_index=255)   #交叉熵损失
            #lovasz_loss = L.lovasz_softmax(out_put, labels, ignore=255)   # lovasz最大值损失
            num_classes = out_put1.shape[1]
            # 计算 Dice 损失
            dice_loss = D.dice_loss(out_put, labels, num_classes, ignore_index=255)
            
            #focall
            #main_loss = ce_loss_1 + 0.75 * lovasz_loss
            main_loss = ce_loss_1 +  dice_loss
            final_loss = main_loss

            final_loss.backward()
            self.optim.step()
            #if (itera + 1) % 50 == 0:
            if (itera + 1) % 400 == 0:
                print(f'iter is {itera + 1}, overall loss is {final_loss}')
                if (itera + 1) % 800 == 0:
                    self.deep_model.eval()                                                                
                    rec, pre, oa, f1_score, iou, kc = self.validation()
                    if f1_score > best_f1_score:
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model.pth'))
                        best_f1_score = f1_score
                        best_round = [rec, pre, oa, f1_score, iou, kc]
                    self.deep_model.train()

            labels = labels.unsqueeze(dim=1)
            if (itera + 1) % 40 == 0:
                vis_image_real_A = make_numpy_grid(de_norm(pre_change_imgs))   #
                vis_image_real_B = make_numpy_grid(de_norm(post_change_imgs))
                vis_gt = make_numpy_grid(labels)
                vis_pre = make_numpy_grid(out_put2* 255)
                vis = np.concatenate(
                    [vis_image_real_A, vis_image_real_B, vis_gt, vis_pre], axis=0)
                vis = np.clip(vis, a_min=0.0, a_max=1.0)
                # file_name = os.path.join("./save/vis/WHU/BCD/train/%d" % (self.epoch),
                #                             'is_training_' + '_' + str(self.epoch) + '_' + str(
                #                             itera + 1) + '.png')
                #file_name = os.path.join(os.makedirs(f"/mnt/workspace/MambaCD/save/bcd/LEVIR-CD256/{self.epoch}ite", exist_ok=True), f"{self.epoch}_{itera + 1}.png")
                # 创建目录
                dir_path = f"/mnt/workspace/MambaCD/save/bcd/LEVIR-CD256/1119_{self.epoch}ite"
                os.makedirs(dir_path, exist_ok=True)

                # 构建文件名
                file_name = os.path.join(dir_path, f"{self.epoch}_{itera + 1}.png")
                plt.imsave(file_name, vis)


        print('The accuracy of the best round is ', best_round)


        
    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator.reset()
        dataset = ChangeDetectionDatset(self.args.val_dataset_path, self.args.val_data_name_list, 256, None, 'val')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()

        for itera, data in enumerate(val_data_loader):
            pre_change_imgs, post_change_imgs, labels, _ = data
            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()
            #print("gt", labels.shape)           #val:gt torch.Size([1, 256, 256, 3]),,,tst:torch.Size([1, 256, 256])
            #labels = torch.sum(labels, dim=-1, keepdim=False)
            #print('gt', labels.shape)  #torch.Size([1, 256, 256])
            #print('pre', pre_change_imgs.shape)   #torch.Size([1, 3, 256, 256])
            output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

            output_1 = output_1.data.cpu().numpy()
            output_1 = np.argmax(output_1, axis=1)
            #print(output_1.shape)   #(1, 256, 256)
            labels = labels.cpu().numpy()

            self.evaluator.add_batch(labels, output_1)
        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'Racall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')
        return rec, pre, oa, f1_score, iou, kc


def main():
    parser = argparse.ArgumentParser(description="Training on  dataset")
    #parser.add_argument('--cfg', type=str, default='home/ywd/wmt/MambaCD/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument('--cfg', type=str, default='/mnt/workspace/MambaCD/changedetection_c/configs/vssm1/vssm_small_224.yaml')
   
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str, default='/mnt/workspace/MambaCD/pretrained_weight/vssm_small_0229_ckpt_epoch_222.pth')
    parser.add_argument('--dataset', type=str, default='LEVIR-CD256')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='/mnt/workspace/datasets/LEVIR-CD256/train')
    parser.add_argument('--train_data_list_path', type=str, default='/mnt/workspace/datasets/LEVIR-CD256/train_list.txt')
    parser.add_argument('--val_dataset_path', type=str, default='/mnt/workspace/datasets/LEVIR-CD256/val')
    parser.add_argument('--val_data_list_path', type=str, default='/mnt/workspace/datasets/LEVIR-CD256/val_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--val_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=False)
    #parser.add_argument('--max_iters', type=int, default=160000)
    parser.add_argument('--max_iters', type=int, default=160000)
    parser.add_argument('--model_type', type=str, default='MambaBCD')
    parser.add_argument('--model_param_path', type=str, default='/mnt/workspace/MambaCD/changedetection_c/saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)


    print(sys.argv)


    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.val_data_list_path, "r") as f:
        # data_name_list = f.read()
        val_data_name_list = [data_name.strip() for data_name in f]
    args.val_data_name_list = val_data_name_list

    trainer = Trainer(args)
    trainer.training()

if __name__ == "__main__":
    main()




