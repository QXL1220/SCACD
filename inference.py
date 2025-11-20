import sys
sys.path.append('/mnt/workspace/MambaCD')

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import imageio
from changedetection_c.configs.config import get_config
from changedetection_c.datasets.make_data_loader import ChangeDetectionDatset
from changedetection_c.models.MambaBCD import STMambaBCD
from changedetection_c.utils_func.metrics import Evaluator

class Inference(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)
        self.evaluator = Evaluator(num_class=2)

        self.deep_model = STMambaBCD(
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
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
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        ) 
        self.deep_model = self.deep_model.cuda()
        #self.model_save_path = os.path.join(args.model_param_path, args.dataset,'change_map/crossself_attention_vss_f1f2_new_str2_ssc')

        self.change_map_saved_path = f"/mnt/workspace/MambaCD/changedetection_c/results/WHU-CD1/318_crossself_attention_vss_new_str2_ssc"
        os.makedirs(self.change_map_saved_path, exist_ok=True)
        
        
        #self.change_map_saved_path = os.path.join(args.result_saved_path, args.dataset, "val——crossself_attention_vss_new_str2_ssc")
        #os.makedirs(self.change_map_saved_path, exist_ok=True)

        # self.change_map_saved_path = os.path.join(args.result_saved_path, args.dataset, 'crossself_attention_vss_f1f2_new_str2_ssc')
        # if not os.path.exists(self.change_map_saved_path):
        #     os.makedirs(self.change_map_saved_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            self.deep_model.load_state_dict(checkpoint)

        self.deep_model.eval()

    def infer(self):
        torch.cuda.empty_cache()
        dataset = ChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        #判断GT中的正样本率，正常不应该为0
        positive_count = 0
        total_pixels = 0
        for data in dataset:
            _, _, label, _ = data
            positive_count += np.sum(label == 1)
            total_pixels += label.size
        print(f"Positive pixel ratio: {positive_count / total_pixels:.4f}")


        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        self.evaluator.reset()

        for itera, data in enumerate(tqdm(val_data_loader, ncols=50)):
            pre_change_imgs, post_change_imgs, labels, names = data
            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda().float()
            labels = labels.cuda().long()

            with torch.no_grad():
                output_1 = self.deep_model(pre_change_imgs, post_change_imgs)
                prob = torch.softmax(output_1, dim=1)
                # print("Prediction probability:", prob.mean().item())  # 输出预测概率均值,,,Prediction probability: 0.5
                predicted_class = torch.argmax(prob, dim=1)  # 获取预测类别
                print("预测结果中正类像素比例：", (predicted_class == 1).float().mean().item())

            output_1 = output_1.data.cpu().numpy()
            output_1 = np.argmax(output_1, axis=1)
            labels = labels.cpu().numpy()

            self.evaluator.add_batch(labels, output_1)

            binary_change_map = np.squeeze(output_1)
            binary_change_map[binary_change_map == 1] = 255
            binary_change_map[binary_change_map == 0] = 0

            image_name = names[0][0:-4] + '.png'
            imageio.imwrite(os.path.join(self.change_map_saved_path, image_name), binary_change_map.astype(np.uint8))

        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'Recall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')

        print('Inference stage is done!')

def main():
    parser = argparse.ArgumentParser(description="Training on WHU-CD1 dataset")
    parser.add_argument('--cfg', type=str, default='/mnt/workspace/MambaCD/changedetection_c/configs/vssm1/vssm_small_224.yaml')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+')
    parser.add_argument('--pretrained_weight_path', type=str, default='/mnt/workspace/MambaCD/pretrained_weight/vssm_small_0229_ckpt_epoch_222.pth')
    parser.add_argument('--dataset', type=str, default='WHU-CD1')
    parser.add_argument('--test_dataset_path', type=str, default='/mnt/workspace/datasets/WHU-CD1/test')
    parser.add_argument('--test_data_list_path', type=str, default='/mnt/workspace/datasets/WHU-CD1/test_list.txt')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--max_iters', type=int, default=160000)
    parser.add_argument('--model_type', type=str, default='MambaBCD_Small')
    parser.add_argument('--result_saved_path', type=str, default='/mnt/workspace/wmt/MambaCD/changedetection_c/results')
    parser.add_argument('--resume', type=str, default='/mnt/workspace/MambaCD/changedetection_c/saved_models/WHU-CD1/317_crossself_attention_youvss_f1f2_new_str2_ssc/21200_model.pth')

    args = parser.parse_args()

    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    infer = Inference(args)
    infer.infer()

if __name__ == "__main__":
    main()



# import sys
# sys.path.append('/mnt/workspace/MambaCD')

# import argparse
# import os
# import time

# import numpy as np

# from changedetection_c.configs.config import get_config
# from torchvision import utils
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from changedetection_c.datasets.make_data_loader import ChangeDetectionDatset, make_data_loader
# from changedetection_c.utils_func.metrics import Evaluator
# from changedetection_c.models.MambaBCD import STMambaBCD
# import imageio
# import changedetection_c.utils_func.lovasz_loss as L
# import matplotlib.pyplot as plt
# from typing import Any, BinaryIO, List, Optional, Tuple, Union

# # def make_numpy_grid(tensor_data, pad_value=0, padding=0):
# #     if isinstance(tensor_data, torch.Tensor):
# #         tensor_data = tensor_data.detach().cpu()
# #     vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
# #     vis = np.array(vis).transpose((1, 2, 0))
# #     # vis = np.array(tensor_data)
# #     # vis = np.transpose(vis, (1, 2, 0))
# #     if vis.shape[2] == 1:
# #         vis = np.stack([vis, vis, vis], axis=-1)
# #     if vis.shape[2] == 2:
# #         channel = vis[:, :, 1:2]
# #         vis = np.concatenate([channel, channel, channel], axis=2)
# #     return vis

# # def de_norm(tensor_data):
# #     return tensor_data * 0.5 + 0.5



# class Inference(object):
#     def __init__(self, args):
#         self.args = args
#         config = get_config(args)
#         self.feature_maps = []
#         self.evaluator = Evaluator(num_class=2)

#         self.deep_model = STMambaBCD(
#             pretrained=args.pretrained_weight_path,
#             patch_size=config.MODEL.VSSM.PATCH_SIZE, 
#             in_chans=config.MODEL.VSSM.IN_CHANS, 
#             num_classes=config.MODEL.NUM_CLASSES, 
#             depths=config.MODEL.VSSM.DEPTHS, 
#             dims=config.MODEL.VSSM.EMBED_DIM, 
#             # ===================
#             ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
#             ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
#             ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
#             ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
#             ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
#             ssm_conv=config.MODEL.VSSM.SSM_CONV,
#             ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
#             ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
#             ssm_init=config.MODEL.VSSM.SSM_INIT,
#             forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
#             # ===================
#             mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
#             mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
#             mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
#             # ===================
#             drop_path_rate=config.MODEL.DROP_PATH_RATE,
#             patch_norm=config.MODEL.VSSM.PATCH_NORM,
#             norm_layer=config.MODEL.VSSM.NORM_LAYER,
#             downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
#             patchembed_version=config.MODEL.VSSM.PATCHEMBED,
#             gmlp=config.MODEL.VSSM.GMLP,
#             use_checkpoint=config.TRAIN.USE_CHECKPOINT,
#             ) 
#         self.deep_model = self.deep_model.cuda()
#         self.epoch = args.max_iters // args.batch_size

#         self.change_map_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'crossattention_change_map')
#         #self.change_map_saved_path = f"/mnt/workspace/wmt/MambaCD/changedetection_c/results/WHU-CD1/change_map/{self.epoch}ite"

#         self.feature_map_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'feature_map')
        
#         if not os.path.exists(self.change_map_saved_path):
#             os.makedirs(self.change_map_saved_path)

#         # self.model_save_path = os.path.join(args.model_param_path, args.dataset,
#         #                                     args.model_type + '_' + datetime.datetime.now().strftime("%m_%d_%H"))
#         # vis_image_real_A = make_numpy_grid(de_norm(pre_change_imgs))   #
#         # vis_image_real_B = make_numpy_grid(de_norm(post_change_imgs))
#         # vis_gt = make_numpy_grid(labels)
#         # vis_pre = make_numpy_grid(out_put2* 255)
#         # vis = np.concatenate(
#         #     [vis_image_real_A, vis_image_real_B, vis_gt, vis_pre], axis=0)
#         # vis = np.clip(vis, a_min=0.0, a_max=1.0)

#         # dir_path = f"/mnt/workspace/wmt/MambaCD/changedetection_c/results/WHU-CD1/change_map/{self.epoch}ite"
#         # os.makedirs(dir_path, exist_ok=True)

#         # #file_name = os.path.join(dir_path, f"{self.epoch}_{itera + 1}.png")
#         # plt.imsave(file_name, vis)

#         # def hook_fn(module, input, output):
#         #     self.feature_maps.append(output.detach().cpu())
#         # self.deep_model.layer3.register_forward_hook(hook_fn)

#         if not os.path.exists(self.feature_map_saved_path):
#             os.makedirs(self.feature_map_saved_path)
#         # 加载检查点
#         if args.resume is not None:
#             if not os.path.isfile(args.resume):
#                 raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
#             checkpoint = torch.load(args.resume)
#             model_dict = {}
#             state_dict = self.deep_model.state_dict()
#             for k, v in checkpoint.items():
#                 if k in state_dict:
#                     model_dict[k] = v
#             state_dict.update(model_dict)
#             self.deep_model.load_state_dict(state_dict)

#         self.deep_model.eval()


#     def infer(self):
#         torch.cuda.empty_cache()
#         dataset = ChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
#         val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
#         torch.cuda.empty_cache()
#         self.evaluator.reset()
            
#         vbar = tqdm(val_data_loader, ncols=50)
#         for itera, data in enumerate(val_data_loader):
#             pre_change_imgs, post_change_imgs, labels, names = data
#             #labels = torch.from_numpy(labels)
#             pre_change_imgs = pre_change_imgs.cuda().float()  #torch.Size([1, 3, 1024, 1024])
#             post_change_imgs = post_change_imgs.cuda()
#             labels = labels.cuda().long()

#             output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

#             output_1 = output_1.data.cpu().numpy()  #(1, 2, 1024, 1024)
#             output_1 = np.argmax(output_1, axis=1)  #(1, 1024, 1024)
#             labels = labels.cpu().numpy()
           
#             self.evaluator.add_batch(labels, output_1) #生成标签和output的混淆矩阵

#         #     vis_image_real_A = make_numpy_grid(de_norm(pre_change_imgs))   #
#         #     vis_image_real_B = make_numpy_grid(de_norm(post_change_imgs))
#         #     labels = np.array(labels)  # 您的 NumPy 数组数据

#         #  # 将 numpy.ndarray 转换为 torch.Tensor
#         #     labels = torch.from_numpy(labels)
#         #     labels = labels.unsqueeze(dim=1)
            
#         #     vis_gt = make_numpy_grid(labels)
#         #     vis_pre = make_numpy_grid(output_1 * 255)
#         #     vis = np.concatenate(
#         #             [vis_image_real_A, vis_image_real_B, vis_gt, vis_pre], axis=0)
#         #     vis = np.clip(vis, a_min=0.0, a_max=1.0)

#             binary_change_map = np.squeeze(output_1)   #(1024, 1024)   max:1
#             binary_change_map[binary_change_map==1] = 255   #(1024, 1024)  max:255
#             binary_change_map[binary_change_map == 0] = 0    # 未变化区域设为 0
#             image_name = names[0][0:-4] + f'.png'
#             imageio.imwrite(os.path.join(self.change_map_saved_path, image_name), binary_change_map.astype(np.uint8))# 将二进制变化图保存为图像文件
#                     # 保存特征图
#         for i, feat in enumerate(self.feature_maps):
#             feat_np = feat.squeeze().numpy()  # [C, H, W]
#                 # 归一化到 0-255
#             feat_np = (feat_np - feat_np.min()) / (feat_np.max() - feat_np.min()) * 255
#             feat_np = feat_np.transpose(1, 2, 0).astype(np.uint8)
#             imageio.imwrite(
#                 os.path.join(self.feature_map_saved_path, f'{names[0]}_feat{i}.png'),
#                 feat_np
#                 )
       

#         f1_score = self.evaluator.Pixel_F1_score()
#         oa = self.evaluator.Pixel_Accuracy()
#         rec = self.evaluator.Pixel_Recall_Rate()
#         pre = self.evaluator.Pixel_Precision_Rate()
#         iou = self.evaluator.Intersection_over_Union()
#         kc = self.evaluator.Kappa_coefficient()
#         print(f'Racall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
#               f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')         
    
#         print('Inference stage is done!')
            


# def main():
#     parser = argparse.ArgumentParser(description="Training on WHU-CD1 dataset")
#     parser.add_argument('--cfg', type=str, default='/mnt/workspace/MambaCD/changedetection_c/configs/vssm1/vssm_small_224.yaml')
#     parser.add_argument(
#         "--opts",
#         help="Modify config options by adding 'KEY VALUE' pairs. ",
#         default=None,
#         nargs='+',
#     )
#     parser.add_argument('--pretrained_weight_path', type=str,default ='/mnt/workspace/MambaCD/pretrained_weight/vssm_small_0229_ckpt_epoch_222.pth' )
#     parser.add_argument('--dataset', type=str, default='WHU-CD1')
#     parser.add_argument('--test_dataset_path', type=str, default='/mnt/workspace/datasets/WHU-CD1/test')
#     parser.add_argument('--test_data_list_path', type=str, default='/mnt/workspace/datasets/WHU-CD1/test_list.txt')
#     parser.add_argument('--batch_size', type=int, default=4)
#     parser.add_argument('--crop_size', type=int, default=256)
#     parser.add_argument('--train_data_name_list', type=list)
#     parser.add_argument('--test_data_name_list', type=list)
#     parser.add_argument('--start_iter', type=int, default=0)
#     parser.add_argument('--cuda', type=bool, default=True)
#     parser.add_argument('--max_iters', type=int, default=160000)
#     parser.add_argument('--model_type', type=str, default='MambaBCD_Small')
#     parser.add_argument('--result_saved_path', type=str, default='/mnt/workspace/wmt/MambaCD/changedetection_c/results')

#     parser.add_argument('--resume', type=str,default ='/mnt/workspace/MambaCD/changedetection_c/saved_models/WHU-CD1/crossattention_vss_nof1f2_new_str2_ssc/34400_model.pth')   #将resume后面的值解析为字符串形式

#     args = parser.parse_args()

#     with open(args.test_data_list_path, "r") as f:
#         # data_name_list = f.read()
#         test_data_name_list = [data_name.strip() for data_name in f]
#     args.test_data_name_list = test_data_name_list

#     infer = Inference(args)
#     infer.infer()



# if __name__ == "__main__":
#     main()