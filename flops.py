import torch
from changedetection_c.models.MambaBCD import STMambaBCD
from changedetection_c.models.ChangeDecoder import ChangeDecoder
import torch
from thop import profile
from thop import clever_format

# 创建网络实例
import torch
import torch.nn as nn

# 假设您已经有了STMambaBCD类的定义

# 创建STMambaBCD实例时提供必需的参数

model = STMambaBCD(
    pretrained=False,
    norm_layer='bn',  # 使用Batch Normalization作为规范化层
    ssm_act_layer='silu',  # 使用SiLU作为SSM激活层
    mlp_act_layer='gelu',  # 使用GELU作为MLP激活层
    ssm_d_state=16,  # 假设这是一个整型参数
    ssm_ratio=2.0,  # 假设这是一个浮点型参数
    ssm_dt_rank="auto",  # 假设这是一个字符串参数
    ssm_conv=3,  # 假设这是一个整型参数
    ssm_conv_bias=True,  # 假设这是一个布尔型参数
    ssm_drop_rate=0.0,  # 假设这是一个浮点型参数
    ssm_init='v0',
    forward_type="v2",
    mlp_ratio=4.0,
        mlp_drop_rate= 0.0,
        gmlp=False,
        use_checkpoint= False,
        post_norm = False)  # 假设这是一个字符串参数)

# 确保这些参数与您的模型定义和需求相匹配

# 创建输入张量
x1 = torch.randn(1, 3, 256, 256)
x2 = torch.randn(1, 3, 256, 256)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x1 = x1.to(device)
x2 = x2.to(device)
# 计算 FLOPs 和参数量：计算的包括可训练参数和不可训练参数，用于评估模型的整体复杂度
flops, params = profile(model, inputs=(x1, x2))

# 格式化输出
flops, params = clever_format([flops, params], "%.4f")
print(f"FLOPs: {flops}, Params: {params}")

flops, params = profile(model, inputs=(x1, x2), verbose=False)
print(f"FLOPs: {flops},Params: {params},")

##############################################################################################

# 计算模型的参数量:只计算模型中可训练的参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

params = count_parameters(model)
print(f"模型参数量: {params}")
##############################################################################################
# from torchsummary import summary
# input_shapes = [(3, 256, 256), (3, 256, 256)]
# summary(model, input_shapes)  # 输出参数量