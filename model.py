# """
# original code from facebook research:
# https://github.com/facebookresearch/ConvNeXt
# """
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# LN 计算每个样本所有通道的均值和方差进行归一化
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# 每个stage的基础块 由DW卷积的7x7，1x1，1x1卷积组成
class Block(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        # 深度可分离卷积，不改变特征图大小
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        # 逐点卷积 1x1卷积用nn.Linear实现 通道数扩增四倍 升维
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        # 激活函数
        self.act = nn.GELU()
        # 逐点卷积 降维
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor: # [64, 128, 14, 14]

        shortcut = x # 残差路径

        x = self.dwconv(x)

        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]

        x = self.norm(x)

        x = self.pwconv1(x) # 升维

        x = self.act(x)

        x = self.pwconv2(x) # 还原维度

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)

        return x


# 每个stage的基础块 由DW卷积的7x7，1x1，1x1卷积组成
class Block_FIN(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        # 深度可分离卷积，不改变特征图大小
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        # 逐点卷积 1x1卷积用nn.Linear实现 通道数扩增四倍 升维
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        # 激活函数
        self.act = nn.GELU()
        # 逐点卷积 降维
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 空间池化
            nn.Conv2d(dim, dim // 2, kernel_size=1),  # 替代FC层
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, kernel_size=1),  # 替代FC层
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # [64, 128, 14, 14]
        shortcut = x # 残差路径

        x = self.dwconv(x)

        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]

        x = self.norm(x)

        x = self.pwconv1(x) # 升维

        x = self.act(x)

        x = self.pwconv2(x) # 还原维度

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = self.se(shortcut)*shortcut + self.drop_path(x)

        return x


# class ConvNeXt(nn.Module):
#     r""" ConvNeXt
#         A PyTorch impl of : `A ConvNet for the 2020s`  -
#           https://arxiv.org/pdf/2201.03545.pdf
#     Args:
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
#         dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
#         drop_path_rate (float): Stochastic depth rate. Default: 0.
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#         head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
#     """
#     def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = None,
#                  dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
#                  head_init_scale: float = 1.):
#         super().__init__()
#         self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
#         stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
#                              LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
#         self.downsample_layers.append(stem)
#
#         # 对应stage2-stage4前的3个downsample
#         for i in range(3):
#             downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
#                                              nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
#             self.downsample_layers.append(downsample_layer)
#
#         self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
#         dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#         cur = 0
#         # 构建每个stage中堆叠的block
#         for i in range(4):
#             stage = nn.Sequential(
#                 *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
#                   for j in range(depths[i])]
#             )
#             self.stages.append(stage)
#             cur += depths[i]
#
#         self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
#         self.head = nn.Linear(dims[-1], num_classes)
#         self.apply(self._init_weights)
#         self.head.weight.data.mul_(head_init_scale)
#         self.head.bias.data.mul_(head_init_scale)
#
#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             nn.init.trunc_normal_(m.weight, std=0.2)
#             nn.init.constant_(m.bias, 0)
#
#     def forward_features(self, x: torch.Tensor) -> torch.Tensor:
#         for i in range(4):
#             x = self.downsample_layers[i](x)
#             x = self.stages[i](x)
#
#         return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x

class ConvNeXt(nn.Module):

    def __init__(self, in_chans: int = 3, num_classes: int = 5, depths: list = None, sigma: float = 0.5,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1., omega: float = 10):

        super().__init__()

        self.sigma = sigma
        self.omega = omega
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers

        # 通道数 3->96, k = 4, s = 4, 直接下采样4倍 224->56
        # stage1
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                             ) # 卷积+LN

        self.downsample_layers.append(stem)

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2) # 用步长为2的卷积进行下采样
                                             )

            self.downsample_layers.append(downsample_layer)

        # stage2-state5层 每个stage组成多个block
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # 生成sum(depths)个连续张量

        # 1x1卷积 模拟FC层
        self.att_conv = nn.Conv2d(dims[-1], num_classes, kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.att_conv3 = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)

        # 1x1卷积 模拟FC层
        self.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_gap = nn.AvgPool2d(14)

        self.bn_att3 = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(14, stride=1)  # 这是ResNet中的最后三层
        self.fc = nn.Linear(dims[-1], num_classes)

        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i]) # [3, 3, 9, 3]
                  ] # 取出每个Block，依次放入容器中 代表一个stage的所有块 后续会依次执行
            )

            self.stages.append(stage) # 放入一个stage
            cur += depths[i] # Drop_Path层的深度

        self.stage4 = nn.Sequential(
                *[Block(dim=dims[3], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[3]) # [3, 3, 9, 3]
                  ] # 取出每个Block，依次放入容器中 代表一个stage的所有块 后续会依次执行
            )

        # 特征重要性网络

        self.att_conv1 = nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1, bias=False)
        self.att_ln1 = LayerNorm(dims[-1], data_format="channels_first")
        self.relu1 = nn.ReLU(inplace=True)

        self.att_conv2 = nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1, bias=False)
        self.att_ln2 = LayerNorm(dims[-1], data_format="channels_first")
        self.relu2 = nn.ReLU(inplace=True)

        self.att_conv3 = nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1, bias=False)
        self.att_ln3 = LayerNorm(dims[-1], data_format="channels_first")
        self.relu3 = nn.ReLU(inplace=True)

        self.att_conv4 = nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1, bias=False)
        self.att_ln4 = LayerNorm(dims[-1], data_format="channels_first")
        self.relu4 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(14, stride=1)
        self.avgpool1 = nn.AvgPool2d(14, stride=1)
        self.norm1 = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer 归一化层
        self.head1 = nn.Linear(dims[-1], num_classes)  # 线性映射层

        self.avgpool2 = nn.AvgPool2d(14, stride=1)
        self.norm2 = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer 归一化层
        self.head2 = nn.Linear(dims[-1], num_classes)  # 线性映射层

        self.norm3 = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer 归一化层
        self.head3 = nn.Linear(dims[-1], num_classes)  # 线性映射层

        self.apply(self._init_weights) # self.apply函数用于递归地将一个函数应用于nn.Module中的所有子模块 这个函数通常用于初始化模型的权重

        self.head1.weight.data.mul_(head_init_scale) # 初始化head层
        self.head1.bias.data.mul_(head_init_scale)

        self.head2.weight.data.mul_(head_init_scale)  # 初始化head层
        self.head2.bias.data.mul_(head_init_scale)

        self.se1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 空间池化
            nn.Conv2d(dims[-1], dims[-1] // 2, kernel_size=1), # 替代FC层
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[-1] // 2, dims[-1], kernel_size=1), # 替代FC层
            nn.Sigmoid()
        )

        self.se2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 空间池化
            nn.Conv2d(dims[-1], dims[-1] // 2, kernel_size=1),  # 替代FC层
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[-1] // 2, dims[-1], kernel_size=1),  # 替代FC层
            nn.Sigmoid()
        )

        self.se3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 空间池化
            nn.Conv2d(dims[-1], dims[-1] // 2, kernel_size=1),  # 替代FC层
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[-1] // 2, dims[-1], kernel_size=1),  # 替代FC层
            nn.Sigmoid()
        )

        self.se4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 空间池化
            nn.Conv2d(dims[-1], dims[-1] // 4, kernel_size=1),  # 替代FC层
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[-1] // 4, dims[-1], kernel_size=1),  # 替代FC层
            nn.Sigmoid()
        )

        self.stage4_FIN = nn.Sequential(
            *[Block_FIN(dim=dims[3], layer_scale_init_value=layer_scale_init_value)
              for j in range(8)  # [2, 2, 2, 2] depths[3]
              ]  # 取出每个Block，依次放入容器中 代表一个stage的所有块 后续会依次执行
        )

        self.down_ln = LayerNorm(dims[-2], eps=1e-6, data_format="channels_first")
        self.down_conv = nn.Conv2d(dims[-2], dims[-1], kernel_size=3, stride=1, padding=1)  # 用步长为2的卷积进行下采样

    def _init_weights(self, m): # 传入nn.Module中的每个层
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            # nn.init.constant_(m.bias, 0)

    def forward_ABN(self, x):# ConvNext+ABN架构

        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)  # 前三个stage得到的特征 [64, 512, 14, 14]

        fe = x

        # ax = self.downsample_layers[-1](x)
        ax = self.stages[-1](x)  # 注意力分支中的卷积层输出特征图 调整通道数

        ax = self.relu(self.bn_att2(self.att_conv(ax)))  # 1x1卷积 再次调整通道数

        bs, cs, ys, xs = ax.shape

        self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax)))  # 1x1卷积 模拟FC层 得到各个通道的attn权重 # [1, 1, 14, 14]

        # 1.注意力分支 1x1卷积+GAP+Softmax
        ax = self.att_conv2(ax)  # 没有用FC得到attn输出
        ax = self.att_gap(ax)
        ax = ax.view(ax.size(0), -1)  # 展平

        # 2.注意力机制
        # print('x', x.shape) # [1, 192, 14, 14]
        rx = x * self.att # 得到注意力图？
        rx = rx + x

        # 3.感知分支
        rx = self.stage4(rx)  # 再经过第四个stage？
        rx = self.avgpool(rx)
        rx = rx.view(rx.size(0), -1)
        rx = self.fc(rx)  # 感知分支的输出

        # return rx, ax
        return self.att

    def forward_LFICAM(self, x: torch.Tensor): # LFICAM

        # x = self.forward_features(x)
        # x = self.head(x) # 再经过head层 [N, C]-->[N, num_class]

        input = x

        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        ax = self.stages[-1](x)

        b, c, w, h = ax.shape

        ex = ax

        input_gray = torch.mean(input, dim=1, keepdim=True)  # 转为单通道灰度图

        input_resized = F.interpolate(input_gray, (w, h), mode='bilinear')

        fe = ax.clone()

        org = fe.clone()

        a1, a2, a3, a4 = fe.size()
        fe = fe.view(a1, a2, -1)

        fe = fe - fe.min(2, keepdim=True)[0]
        fe = fe / fe.max(2, keepdim=True)[0]
        fe = fe.view(a1, a2, a3, a4)  # [32, 128, 14, 14]

        fe[torch.isnan(fe)] = 1
        fe[(org == 0)] = 0

        new_fe = fe * input_resized  # 对原始图像进行mask 不重要的像素点被抑制接近0

        ax = self.att_conv1(new_fe)  # 3x3 卷积
        ax = self.att_ln1(ax)
        ax = self.relu1(ax) # 引入非线性

        ax = self.att_conv2(ax)
        ax = self.att_ln2(ax)
        ax = self.relu2(ax)

        ax = self.att_conv3(ax)
        ax = self.att_ln3(ax)
        ax = self.relu3(ax)

        ax = self.att_conv4(ax)
        ax = self.att_ln4(ax)

        ax = self.avgpool(ax)  # 再经过空间均值池化得到每个通道权重

        w = F.softmax(ax.view(ax.size(0), -1), dim=1)  # 每个通道经过Softmax求权重

        b, c, u, v = fe.size()  # # [32, 128, 14, 14]
        score_saliency_map = torch.zeros((b, 1, u, v)) # 显著性图 只考虑空间位置 不考虑通道

        for i in range(c):  # 512

            saliency_map = torch.unsqueeze(ex[:, i, :, :], 1)  # ex是stage4的输出 取出每一个通道数据[32, 1, 14, 14]

            score = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(w[:, i], 1), 1), 1)

            score_saliency_map = score_saliency_map + score * saliency_map  # 然后再按通道维度对所有特征图求和 最终得到加权后的特征图 [32, 1, 14, 14]

        score_saliency_map = F.relu(score_saliency_map)  # 再经过relu激活 去除负值
        org = score_saliency_map.clone()  # 原始加权后的特征图 用于给stage4加权 [32, 1, 14, 14]
        a1, a2, a3, a4 = score_saliency_map.size()  # [32, 1, 14, 14]
        score_saliency_map = score_saliency_map.view(a1, a2, -1)  # [32, 1, 196]
        score_saliency_map = score_saliency_map - score_saliency_map.min(2, keepdim=True)[0]  # 对所有像素点权重再归一化[0, 1]之间
        score_saliency_map = score_saliency_map / score_saliency_map.max(2, keepdim=True)[0]
        score_saliency_map = score_saliency_map.view(a1, a2, a3, a4)  # [32, 1, 14, 14]
        score_saliency_map[torch.isnan(score_saliency_map)] = org[torch.isnan(score_saliency_map)]

        att = score_saliency_map  # 得到注意力图 [B, 1, 14, 14]

        rx = att * ex  # 将stage4的输出特征图 和注意力图进行加权 [B, 128, 14, 14] * [B, 1, 14, 14]
        rx = rx + ex  # 注意力机制

        rx = self.norm1(rx.mean([-2, -1])) # 正样本
        rx = self.head1(rx)

        return att

    def forward(self, x: torch.Tensor):

        input = x

        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x) # 四个stage得到的特征 [32, 128, 7, 7]

        ax = self.stages[-1](x) # stage4的输出 没有经过下采样 但还是经过了BN等层
        # print(ax.shape)

        b, c, w, h = ax.shape

        ex = ax # stage4的输出 浅拷贝 引用的是同一个对象

        # 获取灰度图
        input_gray = torch.mean(input, dim=1, keepdim=True)  # 转为单通道灰度图
        # print(f'input_gray: {input_gray.shape}') # [32, 1, 224, 224]

        # 灰度图下采样
        input_resized = F.interpolate(input_gray, (w, h), mode='bilinear')

        fe = ax.clone()  # 对stage4的抽象特征进行归一化到[0, 1]之间 视为mask贴到原图上 这里用克隆不会影响stage4的原始输出

        org = fe.clone()  # 保存改变前的值

        a1, a2, a3, a4 = fe.size()  # [B, C, H, W]
        fe = fe.view(a1, a2, -1)

        fe = fe - fe.min(2, keepdim=True)[0]
        fe = fe / fe.max(2, keepdim=True)[0]
        fe = fe.view(a1, a2, a3, a4)  # [32, 128, 14, 14]

        fe[torch.isnan(fe)] = 1
        fe[(org == 0)] = 0

        new_fe = fe * input_resized  # 对原始图像进行mask 不重要的像素点被抑制接近0

        # ax = self.att_conv1(new_fe)  # 3x3 卷积
        # ax = self.att_ln1(ax)
        # ax = self.relu1(ax)  # 引入非线性

        # ax = self.att_conv2(ax)
        # ax = self.att_ln2(ax)
        # ax = self.relu2(ax)

        # ax = self.att_conv3(ax)
        # ax = self.att_ln3(ax)
        # ax = self.relu3(ax)

        # ax = self.att_conv4(ax)
        # ax = self.att_ln4(ax)

        ax = self.stage4_FIN(ax) # 直接用特征图效果更高 原因？
        # ax = self.stage4_FIN(new_fe) # 这里不用Score-CAM的做法效果更好

        ax = self.avgpool(ax)  # 再经过空间均值池化得到每个通道权重
        # ax = self.avgpool(new_fe)  # 再经过空间均值池化得到每个通道权重

        w = F.softmax(ax.view(ax.size(0), -1), dim=1)  # 每个通道经过Softmax求权重

        b, c, u, v = fe.size()  # # [32, 128, 14, 14]

        score_saliency_map = torch.zeros((b, 1, u, v)).cuda() # 显著性图 只考虑空间位置 不考虑通道

        for i in range(c):  # 512

            saliency_map = torch.unsqueeze(ex[:, i, :, :], 1)  # ex是stage4的输出 取出每一个通道数据[32, 1, 14, 14]

            score = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(w[:, i], 1), 1), 1)

            score_saliency_map = score_saliency_map + score * saliency_map  # 然后再按通道维度对所有特征图求和 最终得到加权后的特征图 [32, 1, 14, 14]

        score_saliency_map = F.relu(score_saliency_map)  # 再经过relu激活 去除负值
        org = score_saliency_map.clone()  # 原始加权后的特征图 用于给stage4加权 [32, 1, 14, 14]
        a1, a2, a3, a4 = score_saliency_map.size()  # [32, 1, 14, 14]
        score_saliency_map = score_saliency_map.view(a1, a2, -1)  # [32, 1, 196]
        score_saliency_map = score_saliency_map - score_saliency_map.min(2, keepdim=True)[0]  # 对所有像素点权重再归一化[0, 1]之间
        score_saliency_map = score_saliency_map / score_saliency_map.max(2, keepdim=True)[0]
        score_saliency_map = score_saliency_map.view(a1, a2, a3, a4)  # [32, 1, 14, 14]
        score_saliency_map[torch.isnan(score_saliency_map)] = org[torch.isnan(score_saliency_map)]

        att = score_saliency_map  # 得到注意力图

        rx = att * ex  # 将stage4的输出特征图 和注意力图进行加权 [B, 128, 14, 14] * [B, 1, 14, 14]
        rx = rx + ex  # 注意力机制

        mask = F.sigmoid(self.omega * (score_saliency_map - self.sigma))
        rx2 = ex - (ex * mask)  # 基本会去掉mask区域 也就是显著性区域

        rx = self.norm1(rx.mean([-2, -1])) # 正样本
        rx = self.head1(rx)

        rx2 = self.norm1(rx2.mean([-2, -1])) # 负样本
        rx2 = self.head1(rx2)

        ax2 = self.norm3(ax.squeeze(-1).squeeze(-1)) # 直接监督信号
        ax2 = self.head3(ax2)

        return rx, rx2, ax2


def convnext_tiny(num_classes: int): # 对标ResNet50
    # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    # model = ConvNeXt(depths=[2, 2, 2, 2],
    #                      dims=[48, 96, 192, 384],
    #                      num_classes=num_classes
    #                      )


    # model = ConvNeXt(depths=[2, 2, 2, 2],
    #                  dims=[48, 96, 192, 192],
    #                  num_classes=num_classes,
    #                  ) # 原始


    model = ConvNeXt(depths=[2, 2, 2, 2],
                     dims=[48, 96, 192, 192],
                     num_classes=num_classes,
                     ) # 原始

    # model = ConvNeXt(depths=[2, 2, 2, 2],
    #                  dims=[12, 24, 48, 48],
    #                  num_classes=num_classes,
    #                  )  # 原始

    # model = ConvNeXt(depths=[3, 4, 6, 3],
    #                  dims=[48, 96, 192, 192],
    #                  num_classes=num_classes,
    #                  )  # 原始

    # model = ConvNeXt(depths=[2, 2, 2, 2],
    #                  dims=[128, 256, 512, 1024],
    #                  num_classes=num_classes,
    #                  )
    return model

def convnext_small(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def convnext_base(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes)
    return model


def convnext_large(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes)
    return model


def convnext_xlarge(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes)
    return model

if __name__ == '__main__':
    data = torch.randn((1, 3, 224, 224)).cuda()
    model = convnext_tiny(5)
    rx = model.cuda()(data)

