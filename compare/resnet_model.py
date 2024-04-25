import torch.nn as nn
import torch
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

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

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 cam=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 32
        self.cam = cam

        self.sigma = 0.5
        self.omega = 10
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 32, blocks_num[0])
        self.layer2 = self._make_layer(block, 64, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 128, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 128, blocks_num[3], stride=2)

        self.att_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.att_ln1 = LayerNorm(128, data_format="channels_first")
        self.relu1 = nn.ReLU()

        self.att_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.att_ln2 = LayerNorm(128, data_format="channels_first")
        self.relu2 = nn.ReLU()

        self.att_conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.att_ln3 = LayerNorm(128, data_format="channels_first")
        self.relu3 = nn.ReLU()

        self.att_conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.att_ln4 = LayerNorm(128, data_format="channels_first")
        self.relu4 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool1 = nn.AvgPool2d(7, stride=1)
        self.norm1 = nn.LayerNorm(128, eps=1e-6)  # final norm layer 归一化层
        self.head1 = nn.Linear(128, num_classes)  # 线性映射层

        self.avgpool2 = nn.AvgPool2d(7, stride=1)
        self.norm2 = nn.LayerNorm(128, eps=1e-6)  # final norm layer 归一化层
        self.head2 = nn.Linear(128, num_classes)  # 线性映射层

        self.norm3 = nn.LayerNorm(128, eps=1e-6)  # final norm layer 归一化层
        self.head3 = nn.Linear(128, num_classes)  # 线性映射层

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):

        if not self.cam:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            per = x
            if self.include_top:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x, per
        else:
            input = x

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            ax = self.layer4(x)

            ex = ax  # stage4的输出

            # 获取灰度图
            input_gray = torch.mean(input, dim=1, keepdim=True)  # 转为单通道灰度图
            # print(f'input_gray: {input_gray.shape}') # [32, 1, 224, 224]

            # 灰度图下采样
            input_resized = F.interpolate(input_gray, (7, 7), mode='bilinear')

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

            ax = self.att_conv1(new_fe)  # 3x3 卷积
            ax = self.att_ln1(ax)
            # ax = self.se1(ax) * ax + ax
            ax = self.relu1(ax)

            ax = self.att_conv2(ax)
            ax = self.att_ln2(ax)
            # ax = self.se2(ax) * ax + ax
            ax = self.relu2(ax)

            ax = self.att_conv3(ax)
            ax = self.att_ln3(ax)
            ax = self.relu3(ax)

            ax = self.att_conv4(ax)
            ax = self.att_ln4(ax)
            # ax = self.se4(ax) * ax + ax
            ax = self.relu4(ax)

            ax = self.avgpool(ax)  # 再经过空间均值池化得到每个通道权重

            w = F.softmax(ax.view(ax.size(0), -1), dim=1)  # 每个通道经过Softmax求权重

            b, c, u, v = fe.size()  # # [32, 128, 14, 14]
            score_saliency_map = torch.zeros((b, 1, u, v)).cuda()  # 显著性图 只考虑空间位置 不考虑通道

            for i in range(c):  # 512

                # 得到stage4中每个样本每个通道的数据
                saliency_map = torch.unsqueeze(ex[:, i, :, :], 1)  # ex是stage4的输出 取出每一个通道数据[32, 1, 14, 14]
                score = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(w[:, i], 1), 1), 1)
                # 用权重对每个通道进行加权 汇集了所有通道加权后的结果
                # score_saliency_map += score * saliency_map  # 然后再按通道维度对所有特征图求和 最终得到加权后的特征图[32, 1, 14, 14]
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
            # rx2 = ex - att * ex

            # classifier [N, C]
            rx = self.norm1(rx.mean([-2, -1]))  # 正样本
            rx = self.head1(rx)

            rx2 = self.norm2(rx2.mean([-2, -1]))  # 负样本
            rx2 = self.head2(rx2)

            ax2 = self.norm3(ax.squeeze(-1).squeeze(-1))  # 直接监督信号
            ax2 = self.head3(ax2)

            return rx, rx2, ax2  # 返回两个分支预测结果


def resnet18(num_classes=1000, cam=True, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top, cam=cam)

def resnet34(num_classes=1000, cam=True, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, cam=cam, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, cam=False, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
