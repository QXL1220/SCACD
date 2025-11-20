import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
#from classification.models.local_vmamba import VSSM, VSSBlock, Permute
#from classification.models.spatialmamba import  SpatialMamba, VSSBlock, Permute

class ChangeDecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(ChangeDecoder, self).__init__()

        self.vssblock = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=128, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),)
        
        self.conv_768 = nn.Conv2d(in_channels=768, out_channels=128, kernel_size=1)
        self.conv_384 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1)
        self.conv_192 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1)
        self.conv_96 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=1)
        #elf.conv_final = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)

        self.crossattention = CrossAttention(num_heads=8, qk_scale=None, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, dim=128)
        #self.crossself_attention = CrossSelfAttention(num_heads=8,qk_scale= None,qkv_bias=None, attn_drop=0.0, proj_drop=0.0, dim=128)
    #     self.crossself_vss_attention = CrossSelfAttentionVSS(dim=128,
    # channel_first=False,
    # norm_layer=nn.LayerNorm,
    # ssm_act_layer=nn.GELU,
    # mlp_act_layer=nn.GELU,
    # num_heads=8,
    # qkv_bias=False,
    # qk_scale=None,
    # attn_drop=0.0,
    # proj_drop=0.0,
    # sr_ratio=2,
    # ssm_d_state=True,  # 确保提供这个参数
    # ssm_ratio=0.5,
    # ssm_dt_rank=1,
    # ssm_conv=True,
    # ssm_conv_bias=True,
    # ssm_drop_rate=0.1,
    # ssm_init='xavier',
    # forward_type='normal',
    # mlp_ratio=4.0,
    # mlp_drop_rate=0.0,
    # gmlp=False,
    # use_checkpoint=False)

        # self.selfattention = SelfAttention(num_heads=8, qk_scale=None, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, dim=128)
        
        # self.dualattention = DualAttention(num_heads=8, qk_scale=None, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, dim=128)

        # self.noattention = NoAttention()
        self.fuse_layer = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 2, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 3, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())       
        # Fuse layer    
        # self.fuse_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
        #                                   nn.BatchNorm2d(128), nn.ReLU())
        # self.fuse_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
        #                                   nn.BatchNorm2d(128), nn.ReLU())
        # self.fuse_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
        #                                   nn.BatchNorm2d(128), nn.ReLU())
        # self.fuse_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
        #                                   nn.BatchNorm2d(128), nn.ReLU())
        
        # self.fuse1_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 , out_channels=128),
        #                                   nn.BatchNorm2d(128), nn.ReLU())
        # self.fuse1_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 , out_channels=128),
        #                                   nn.BatchNorm2d(128), nn.ReLU())
        # self.fuse1_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 , out_channels=128),
        #                                   nn.BatchNorm2d(128), nn.ReLU())
        # self.fuse1_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 , out_channels=128),
        #                                   nn.BatchNorm2d(128), nn.ReLU())
        
        self.fusion3 =Spatial_FeatureFusionBlock(256, 128, 3,1)
        self.fusion4 =CBAM_FeatureFusionBlock(256, 128, 3,1)
        # self.fusion5 =NoCBAM(256,128,3,1)

        # Smooth layer
        self.smooth_layer_3 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_2 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_1 = ResBlock(in_channels=128, out_channels=128, stride=1) 



        #fusion layer


    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y
    


#交叉注意力加vss＋原特征拼接+ spatial spatial CBAM
    def _upsample_(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') 
    
    def forward(self, pre_features, post_features):
        
        # 选择设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 将所有特征图移动到指定设备
        pre_features = [feat.to(device) for feat in pre_features]
        post_features = [feat.to(device) for feat in post_features]

        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features
        # print(pre_feat_4.size())# torch.Size([4, 768, 8, 8])
        # print(pre_feat_3.size())#torch.Size([4, 384, 16, 16])
        # print(pre_feat_2.size())# torch.Size([4, 192, 32, 32])
        # print(pre_feat_1.size())# torch.Size([4, 96, 64, 64])

        '''
            Stage I 
        '''
        # print(pre_feat_4.size()) #torch.Size([4, 768, 8, 8])
       # p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))
        #print(p41.size())  #torch.Size([4, 128, 8, 8])
        B, C, H, W = pre_feat_4.size()
        B, C, H, W = pre_feat_4.size()
        pre_feat_4 =self.conv_768(pre_feat_4)
        post_feat_4 = self.conv_768(post_feat_4)
        p4 = self.crossattention(pre_feat_4, pre_feat_4)
        p4 = self.vssblock(p4)
        #p4 = self.fuse_layer3(torch.cat([pre_feat_4, pre_feat_4, p4], dim = 1))
        #print(p4.size())#torch.Size([4, 128, 8, 8])
        '''
            Stage II
        '''
        pre_feat_3 = self.conv_384(pre_feat_3)
        post_feat_3 = self.conv_384(post_feat_3)
        #print("pre 3:", pre_feat_3.size())#[4,128,16,16]
        p3= self.crossattention(pre_feat_3, post_feat_3) #[4,128,16,16]
        p3 = self.vssblock(p3)
        #print("p3",p3.size())

        p4 = self._upsample_(p4, p3)
        #print("p4",p4.size())  #torch.Size([4, 128, 16, 16])
        #print(p3.size())  #torch.Size([4, 128, 16, 16])
        p3 = self.fusion4(p4,p3)
        #print("p3",p3.size())
        #print("p33",p33.size())#([4, 128, 16, 16])     

        #p3 = self.fuse_layer3(torch.cat([pre_feat_3, post_feat_3, p3], dim =1))
        '''
            Stage III
        '''
        pre_feat_2 = self.conv_192(pre_feat_2)
        post_feat_2 = self.conv_192(post_feat_2)
        #print("pre 2:", pre_feat_2.size())#[4,128,32,32]
        p2= self.crossattention(pre_feat_2, post_feat_2)
        p2 = self.vssblock(p2)
        #print(p2.size())

        p3 = self._upsample_(p3, p2)
        #print(p4.size())
        #print(p3.size())
        p2 = self.fusion4(p3,p2)
        #print("p22",p22.size())#([8, 128, 32,32])     

        #p2 = self.fuse_layer3(torch.cat([pre_feat_2, post_feat_2,p2], dim=1))
        '''
            Stage Ⅳ
        '''
        pre_feat_1 = self.conv_96(pre_feat_1)
        post_feat_1 = self.conv_96(post_feat_1)
        #print("pre 1:", pre_feat_1.size())#[4,128,64,64]
        p1= self.crossattention(pre_feat_1, post_feat_1)
        p1 = self.vssblock(p1)
        #print(p1.size())#[4,128,64,64]

        p2 = self._upsample_(p2, p1)
        #print(p4.size())
        #print(p3.size())
        p1 = self.fusion4(p2,p1)
        #print("p11",p11.size())#[4,128,64,64]

        #p1 = self.fuse_layer3(torch.cat([pre_feat_1, post_feat_1,p1], dim=1))
        return p1


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
"""
空间注意力融合
"""
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)   
class Spatial_FeatureFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, kernel=7):
        super(Spatial_FeatureFusionBlock, self).__init__()
        # self.cam = ChannelAttention(in_channels=in_channels)
        self.spatial_attention = SpatialAttention(kernel)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        # self.relu2 = nn.ReLU()
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        # weight = self.cam(x)
        weight=self.spatial_attention(x) * x
        out = weight * x
        out = self.conv1(out)
        out = self.relu1(out)
        # out = self.conv2(out)
        # out = self.relu2(out)
        return out
"""
 通道注意力
 """   
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels, 1, bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)
"""
CBAM融合
"""
class CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class CBAM_FeatureFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, kernel=7):
        super(CBAM_FeatureFusionBlock, self).__init__()
        # self.cam = ChannelAttention(in_channels=in_channels)
        #self.spatial_attention = SpatialAttention(kernel)
        self.cbam  = CBAM(in_channels, ratio=16, kernel_size=7)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        # self.relu2 = nn.ReLU()
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        # weight = self.cam(x)
        weight=self.cbam(x) * x
        out = weight * x
        out = self.conv1(out)
        out = self.relu1(out)
        # out = self.conv2(out)
        # out = self.relu2(out)
        #print(out.shape)torch.Size([4, 128, 16, 16])
        return out
    



'''
无CBAM
''' 
# class NoCBAM(nn.Module):
#     def __init__(self, in_channels, out_channels,kernel_size, padding):
#         super().__init__()
#         self.con2d = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size,padding=padding)
#         self.relu = nn.ReLU()
#     def forward(self, x1, x2):
#         x = torch.cat((x1, x2), dim =1)
#         x = self.con2d(x)
#         x = self.relu(x)
#         #rint(x.shape)#torch.Size([4, 128, 16, 16])
#         return x
    
"""
含有SRA的交叉注意力＋自注意力：CrossSelfAttention
""" 
# class CrossSelfAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
#         super().__init__()
#         #super(CrossSelfAttention, self).__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.wq = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.wk = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.wv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Conv2d(2 * dim, dim, kernel_size=1)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm = nn.LayerNorm(dim)

        
#     def forward(self, x1, x2):
#         B1, C1, H1, W1 = x1.shape
#         B2, C2, H2, W2 = x2.shape

#         # 确保输入特征的通道数与 dim 一致
#         assert C1 == self.wq.in_channels, "x1 的通道数必须等于 dim"
#         assert C2 == self.wk.in_channels, "x2 的通道数必须等于 dim"

#         # 计算 Query
#         q1 = self.wq(x1).view(B1, self.num_heads, C1 // self.num_heads, H1 * W1).permute(0, 1, 3, 2)
#         q2 = self.wq(x2).view(B2, self.num_heads, C2 // self.num_heads, H2 * W2).permute(0, 1, 3, 2)
#         #print(q1.size())#torch.Size([4, 8, 64, 16])
#         if self.sr_ratio > 1:
#             x1_sr = self.sr(x1)
#             x1_sr = x1_sr.permute(0, 2, 3, 1)
#             x1_sr = self.norm(x1_sr)
#             x1_sr = x1_sr.permute(0, 3, 1, 2)
#             k1 = self.wk(x1_sr)
#             v1 = self.wv(x1_sr)
#         else:
#             k1 = self.wk(x1)
#             v1 = self.wv(x1)
#         k1 = k1.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)
#         v1 = v1.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)
#         # print("k1",k1.shape)
#         # print("v1",v1.shape)
#         if self.sr_ratio > 1:
#             x2_sr = self.sr(x2)
#             x2_sr = x2_sr.permute(0, 2, 3, 1)
#             x2_sr = self.norm(x2_sr)
#             x2_sr = x2_sr.permute(0, 3, 1, 2)
#             k2 = self.wk(x2_sr)
#             v2 = self.wv(x2_sr)
#         else:
#             k2 = self.wk(x2)
#             v2 = self.wv(x2)
#         k2 = k2.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)
#         v2 = v2.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)


#         # 计算交叉注意力
#         attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
#         attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
#         attn1 = attn1.softmax(dim=-1)
#         attn1 = self.attn_drop(attn1)
#         attn2 = attn2.softmax(dim=-1)
#         attn2 = self.attn_drop(attn2)
#         # print('atten1',attn1.shape)
#         # 应用注意力
#         x = torch.matmul(attn1, v2).permute(0, 1, 3, 2).reshape(B1, C1, H1, W1)
#         y = torch.matmul(attn2, v1).permute(0, 1, 3, 2).reshape(B2, C2, H2, W2)
#         #print(x.shape)  torch.Size([4, 128, 8, 8])

#         # 计算双重自注意力
#         sattn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
#         sattn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
#         sattn1 = sattn1.softmax(dim=-1)
#         sattn1 = self.attn_drop(sattn1)
#         sattn2 = sattn2.softmax(dim=-1)
#         sattn2 = self.attn_drop(sattn2)
#         # 应用注意力
#         sx = torch.matmul(sattn1, v1).permute(0, 1, 3, 2).reshape(B1, C1, H1, W1)  # [B1, C1, H1, W1]
#         sy = torch.matmul(sattn2, v2).permute(0, 1, 3, 2).reshape(B2, C2, H2, W2)  # [B1, C1, H1, W1]

#         x = torch.cat([x, sx], dim=1)

#         y = torch.cat([y, sy], dim=1)


#         x = self.proj(x)
#         y = self.proj(y)
#         x = torch.cat([x, y], dim =1)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x


"""
含有SRA的交叉注意力＋自注意力+分别vss:CrossSelfAttentionVSS
""" 
# class CrossSelfAttentionVSS(nn.Module):
#     def __init__(self, dim, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2, **kwargs):
#         #super().__init__()
#         super(CrossSelfAttentionVSS, self).__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.wq = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.wk = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.wv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Conv2d(2 * dim, dim, kernel_size=1)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm = nn.LayerNorm(dim)
#         self.vssblock = nn.Sequential(
#             nn.Conv2d(kernel_size=1, in_channels=128, out_channels=128),
#             Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
#             VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
#                 ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
#                 ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
#                 forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
#                 gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
#             Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),)
        
#     def forward(self, x1, x2):
#         B1, C1, H1, W1 = x1.shape
#         B2, C2, H2, W2 = x2.shape

#         # 确保输入特征的通道数与 dim 一致
#         assert C1 == self.wq.in_channels, "x1 的通道数必须等于 dim"
#         assert C2 == self.wk.in_channels, "x2 的通道数必须等于 dim"

#         # 计算 Query
#         q1 = self.wq(x1).view(B1, self.num_heads, C1 // self.num_heads, H1 * W1).permute(0, 1, 3, 2)
#         q2 = self.wq(x2).view(B2, self.num_heads, C2 // self.num_heads, H2 * W2).permute(0, 1, 3, 2)
#         #print(q1.size())#torch.Size([4, 8, 64, 16])
#         if self.sr_ratio > 1:
#             x1_sr = self.sr(x1)
#             x1_sr = x1_sr.permute(0, 2, 3, 1)
#             x1_sr = self.norm(x1_sr)
#             x1_sr = x1_sr.permute(0, 3, 1, 2)
#             k1 = self.wk(x1_sr)
#             v1 = self.wv(x1_sr)
#         else:
#             k1 = self.wk(x1)
#             v1 = self.wv(x1)
#         k1 = k1.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)
#         v1 = v1.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)

#         if self.sr_ratio > 1:
#             x2_sr = self.sr(x2)
#             x2_sr = x2_sr.permute(0, 2, 3, 1)
#             x2_sr = self.norm(x2_sr)
#             x2_sr = x2_sr.permute(0, 3, 1, 2)
#             k2 = self.wk(x2_sr)
#             v2 = self.wv(x2_sr)
#         else:
#             k2 = self.wk(x2)
#             v2 = self.wv(x2)
#         k2 = k2.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)
#         v2 = v2.view(B1, self.num_heads, C1 // self.num_heads, -1).permute(0, 1, 3, 2)


#         # 计算交叉注意力
#         attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
#         attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
#         attn1 = attn1.softmax(dim=-1)
#         attn1 = self.attn_drop(attn1)
#         attn2 = attn2.softmax(dim=-1)
#         attn2 = self.attn_drop(attn2)
#         # 应用注意力
#         x = torch.matmul(attn1, v2).permute(0, 1, 3, 2).reshape(B1, C1, H1, W1)
#         y = torch.matmul(attn2, v1).permute(0, 1, 3, 2).reshape(B2, C2, H2, W2)
#         #print(x.shape)  torch.Size([4, 128, 8, 8])

#         # 计算双重自注意力
#         sattn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
#         sattn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
#         sattn1 = sattn1.softmax(dim=-1)
#         sattn1 = self.attn_drop(sattn1)
#         sattn2 = sattn2.softmax(dim=-1)
#         sattn2 = self.attn_drop(sattn2)
#         # 应用注意力
#         sx = torch.matmul(sattn1, v1).permute(0, 1, 3, 2).reshape(B1, C1, H1, W1)  # [B1, C1, H1, W1]
#         sy = torch.matmul(sattn2, v2).permute(0, 1, 3, 2).reshape(B2, C2, H2, W2)  # [B1, C1, H1, W1]
#         # 合并输出
#         x = torch.cat([x, sx], dim=1)
#         x = self.proj(x)
#         x = self.vssblock(x)

#         y = torch.cat([y, sy], dim=1)
#         y = self.proj(y)
#         y = self.vssblock(y)

#         x = torch.cat([x, y], dim =1)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x

'''
单注意力:通道维度上拼接+卷积（in channel=256， out channel =128），放入单一注意力机制中
'''
# class SelfAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         # 定义 Query、Key、Value 的卷积层
#         self.wq = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.wk = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.wv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

#         # 定义 Dropout 和投影层
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Conv2d(dim, dim, kernel_size=1)
#         self.proj_drop = nn.Dropout(proj_drop)

#         # 空间降采样（SRA）
#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm = nn.LayerNorm(dim)
        
#         #卷积
#         self.conv_256 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)

#     def forward(self, x1, x2):
#         B, C, H, W = x1.shape
#         B, C, H, W = x2.shape
#         # 拼接＋卷积
#         x = torch.cat([x1, x2], dim=1)
#         x = self.conv_256(x)
#         #print(x.shape)

#         # 确保输入特征的通道数与 dim 一致
#         assert C == self.wq.in_channels, "输入通道数必须等于 dim"

#         # 计算 Query、Key、Value
#         q = self.wq(x).view(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)  # [B, h, HW, d]
        
#         if self.sr_ratio > 1:
#             # 空间降采样
#             x_sr = self.sr(x)
#             x_sr = x_sr.permute(0, 2, 3, 1)  # [B, H//sr, W//sr, C]
#             x_sr = self.norm(x_sr)
#             x_sr = x_sr.permute(0, 3, 1, 2)  # [B, C, H//sr, W//sr]
#             k = self.wk(x_sr)
#             v = self.wv(x_sr)
#         else:
#             # 无降采样
#             k = self.wk(x)
#             v = self.wv(x)

#         # 重塑 Key 和 Value
#         k = k.view(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)  # [B, h, (H//sr)*(W//sr), d]
#         v = v.view(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)  # [B, h, (H//sr)*(W//sr), d]

#         # 计算注意力权重
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, h, HW, (H//sr)*(W//sr)]
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         # 应用注意力
#         out = torch.matmul(attn, v).permute(0, 1, 3, 2).reshape(B, C, H, W)  # [B, C, H, W]

#         # 投影层
#         out = self.proj(out)
#         out = self.proj_drop(out)

#         return out
    
'''
不含注意力：直接拼接加卷积
'''
# class NoAttention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_256 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
#     def forward(self, x1, x2):
#         x = torch.cat([x1, x2], dim = 1)
#         x = self.conv_256(x)
#         return x
"""
双重自注意力
"""
# class DualAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.wq = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.wk = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.wv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         #self.proj = nn.Conv2d(dim, dim, kernel_size=1)
#         self.proj = nn.Conv2d(2*dim, dim, kernel_size=1)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x1, x2):
#         B1, C1, H1, W1 = x1.shape  # 解析输入张量的形状
#         B2, C2, H2, W2 = x2.shape  # 解析输入张量的形状

#         # 确保输入特征的通道数与 dim 一致
#         assert C1 == self.wq.in_channels, "x1 的通道数必须等于 dim"
#         assert C2 == self.wk.in_channels, "x2 的通道数必须等于 dim"

#         # 计算 Query, Key, Value
#         q1 = self.wq(x1).view(B1, self.num_heads, C1 // self.num_heads, H1 * W1).permute(0, 1, 3, 2)  # [B1, num_heads, H1*W1, head_dim]
#         k2 = self.wk(x2).view(B2, self.num_heads, C2 // self.num_heads, H2 * W2).permute(0, 1, 3, 2)  # [B2, num_heads, H2*W2, head_dim]
#         v2 = self.wv(x2).view(B2, self.num_heads, C2 // self.num_heads, H2 * W2).permute(0, 1, 3, 2)  # [B2, num_heads, H2*W2, head_dim]

#         q2 = self.wq(x2).view(B2, self.num_heads, C2 // self.num_heads, H2 * W2).permute(0, 1, 3, 2)  # [B1, num_heads, H1*W1, head_dim]
#         k1 = self.wk(x1).view(B1, self.num_heads, C1 // self.num_heads, H1 * W1).permute(0, 1, 3, 2)  # [B2, num_heads, H2*W2, head_dim]
#         v1 = self.wv(x1).view(B1, self.num_heads, C1 // self.num_heads, H1 * W1).permute(0, 1, 3, 2)  # [B2, num_heads, H2*W2, head_dim]
#         # 计算注意力
#         attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
#         attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
#         attn1 = attn1.softmax(dim=-1)
#         attn1 = self.attn_drop(attn1)
#         attn2 = attn2.softmax(dim=-1)
#         attn2 = self.attn_drop(attn2)

#         # 应用注意力
#         x = torch.matmul(attn1, v1).permute(0, 1, 3, 2).reshape(B1, C1, H1, W1)  # [B1, C1, H1, W1]
#         y = torch.matmul(attn2, v2).permute(0, 1, 3, 2).reshape(B2, C2, H2, W2)  # [B1, C1, H1, W1]
       

#         x = torch.cat([x, y], dim=1)   #[4, 256, 8, 8]

#         x = self.proj(x)
#         x = self.proj_drop(x)
#         #x = x.view(B1, 2, C1, H1, W1).permute(0, 1, 2, 3, 4).contiguous().view(B1, 2 * C1, H1, W1)
#         x = x.view(B1,C1,H1,W1)
#         #print(x.size())#torch.Size([4, 128, 8, 8])
#         return x
'''
交叉注意力：CrossAttention
'''
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.wk = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.wv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        #self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj = nn.Conv2d(2*dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        B1, C1, H1, W1 = x1.shape  # 解析输入张量的形状
        B2, C2, H2, W2 = x2.shape  # 解析输入张量的形状

        # 确保输入特征的通道数与 dim 一致
        assert C1 == self.wq.in_channels, "x1 的通道数必须等于 dim"
        assert C2 == self.wk.in_channels, "x2 的通道数必须等于 dim"

        # 计算 Query, Key, Value
        q1 = self.wq(x1).view(B1, self.num_heads, C1 // self.num_heads, H1 * W1).permute(0, 1, 3, 2)  # [B1, num_heads, H1*W1, head_dim]
        k2 = self.wk(x2).view(B2, self.num_heads, C2 // self.num_heads, H2 * W2).permute(0, 1, 3, 2)  # [B2, num_heads, H2*W2, head_dim]
        v2 = self.wv(x2).view(B2, self.num_heads, C2 // self.num_heads, H2 * W2).permute(0, 1, 3, 2)  # [B2, num_heads, H2*W2, head_dim]

        q2 = self.wq(x2).view(B2, self.num_heads, C2 // self.num_heads, H2 * W2).permute(0, 1, 3, 2)  # [B1, num_heads, H1*W1, head_dim]
        k1 = self.wk(x1).view(B1, self.num_heads, C1 // self.num_heads, H1 * W1).permute(0, 1, 3, 2)  # [B2, num_heads, H2*W2, head_dim]
        v1 = self.wv(x1).view(B1, self.num_heads, C1 // self.num_heads, H1 * W1).permute(0, 1, 3, 2)  # [B2, num_heads, H2*W2, head_dim]
        # 计算注意力
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)

        # 应用注意力
        x = torch.matmul(attn1, v2).permute(0, 1, 3, 2).reshape(B1, C1, H1, W1)  # [B1, C1, H1, W1]
        y = torch.matmul(attn2, v1).permute(0, 1, 3, 2).reshape(B2, C2, H2, W2)  # [B1, C1, H1, W1]
       

        x = torch.cat([x, y], dim=1)   #[4, 256, 8, 8]

        x = self.proj(x)
        x = self.proj_drop(x)
        #x = x.view(B1, 2, C1, H1, W1).permute(0, 1, 2, 3, 4).contiguous().view(B1, 2 * C1, H1, W1)
        x = x.view(B1,C1,H1,W1)
        #print(x.size())#torch.Size([4, 128, 8, 8])
        return x











        # # 计算注意力分数
        # attn = torch.matmul(q1, k2.transpose(-2, -1)) * self.scale  # [B1, num_heads, H1*W1, H2*W2]
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # # 应用注意力
        # x = torch.matmul(attn, v2).permute(0, 1, 3, 2).reshape(B1, C1, H1, W1)  # [B1, C1, H1, W1]

        # # 投影和 dropout
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # #print("CrossAttention output shape:", x.shape)#([4, 128, 8, 8])
        # return x   
