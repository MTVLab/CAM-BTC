import os
import sys
import json
import pickle
import random
import math
import numpy as np

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.nn.functional as F

def read_split_data(root: str, json_data):
    random.seed(2024)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    survival_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))] # 获取每个文件夹的相对路径名字

    # 排序，保证各平台顺序一致
    survival_class.sort()

    # 用一个字典生成类别名称和对应的索引
    class_indices = dict() # {'xxx':0}
    for k, v in enumerate(survival_class):
        class_indices[v] = k

    # print(class_indices)

    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4) # 将python对象转为json对象
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str) # 将json对象写入文件中

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息

    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息

    every_class_num = []  # 存储每个类别的样本总数

    # print(survival_class)

    # 遍历每个文件夹下的文件
    for cla in survival_class: # BM GBM
        cla_path = os.path.join(root, cla) # ./data/PIC/BM
        cases_path = os.listdir(cla_path) # [BraTS-MET-00002-000, ...]
        images = []

        # print(cla_path)
        # print(cases_path)

        '''
        更新读取数据：按照病例文件夹读，依次读取一个文件夹下的所有图像数据，以病例为单位：
        '''

        for case in cases_path:
            image = os.listdir(os.path.join(cla_path, case))
            image = [os.path.join(cla_path, case, i) for i in image]
            images += image[:]

        # print(images)

        # 获取该类别对应的索引
        image_class = class_indices[cla] # 获取类别索引

        # 记录该类别的样本数量
        every_class_num.append(len(images))

        for imgs_path in images:
            if imgs_path.split('\\')[-2] in json_data:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(imgs_path)  # 直接放入病例下的所有文件列表
                val_images_label.append(image_class)  # 图片文件和标签一一对应
            else:  # 否则存入训练集
                train_images_path.append(imgs_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False

    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(survival_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(survival_class)), survival_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('survival class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

# 训练一个epoch的类
def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, cam):
    model.train() # 开启训练

    loss_function = torch.nn.CrossEntropyLoss() # 交叉熵损失
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_loss_pos = torch.zeros(1).to(device)  # 累计损失
    accu_loss_neg = torch.zeros(1).to(device)  # 累计损失
    accu_loss_sup = torch.zeros(1).to(device)  # 累计损失

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数

    optimizer.zero_grad() # 先是AdamW优化器梯度清零

    sample_num = 0 # 总的训练样本数？

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data

        sample_num += images.shape[0]

        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=5).cuda()

        if cam:
            pred, pred2, pred3 = model(images.to(device))  # 得到预测结果 [N,C]
            loss_neg = F.sigmoid(pred2) * one_hot_labels  # 应该最小化这个损失 也就是labels为1处于该预测值处的损失应该最小 负样本
            loss_neg = loss_neg.sum().float() / one_hot_labels.sum()  # 每个样本平均损失
            loss_pos = loss_function(pred, labels.to(device))  # 自动计算的是每个样本的损失均值 正样本
            loss_sup = loss_function(pred3, labels.to(device))  # 自动计算的是每个样本的损失均值 直接监督损失
            if epoch > 10:
                loss = 0.5 * loss_pos + 0.5 * loss_neg + loss_sup
            else:
                loss = loss_pos + loss_sup

            loss.backward()
            accu_loss_pos += loss_pos.detach()
            accu_loss_neg += loss_neg.detach()
            accu_loss_sup += loss_sup.detach()

        else:
            pred = model(images.to(device)) # 得到预测结果 [N,C]
            loss = loss_function(pred, labels.to(device))
            loss.backward()

        pred_classes = torch.max(pred, dim=1)[1] # [1]是取最大值所在的索引 得到[B, index] [1]取的是索引表示每个样本的预测索引 即标签

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        accu_loss += loss.detach()

        # 作用是用于设置 tqdm 进度条的描述信息
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step() # 优化器更新
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step() # 学习率更新

    # 返回每个epoch训练完毕的loss和每个epoch内平均准确率
    if cam:
        return accu_loss.item() / (step + 1), accu_loss_pos.item() / (step + 1), accu_loss_neg.item() / (
                    step + 1), accu_loss_sup.item() / (step + 1), \
               accu_num.item() / sample_num
    else:
        return accu_loss.item() / (step + 1), accu_num.item() / sample_num

# 验证
@torch.no_grad()
def evaluate(model, data_loader, device, epoch, cam):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval() # 开始验证模式

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_loss_pos = torch.zeros(1).to(device)  # 累计损失
    accu_loss_neg = torch.zeros(1).to(device)  # 累计损失
    accu_loss_sup = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=5).cuda()

        if cam:
            pred, pred2, pred3 = model(images.to(device))  # 得到预测结果 [N,C]
            loss_neg = F.sigmoid(pred2) * one_hot_labels  # 应该最小化这个损失 也就是labels为1处该预测值处的损失应该最小 负样本
            loss_neg = loss_neg.sum().float() / one_hot_labels.sum()  # 每个样本平均损失
            loss_pos = loss_function(pred, labels.to(device))  # 自动计算的是每个样本的损失均值 正样本
            loss_sup = loss_function(pred3, labels.to(device))  # 自动计算的是每个样本的损失均值 直接监督
            if epoch > 5:
                loss = loss_pos + loss_neg + loss_sup
            else:
                loss = loss_pos + loss_sup

            accu_loss_pos += loss_pos.detach()
            accu_loss_neg += loss_neg.detach()
            accu_loss_sup += loss_sup.detach()

        else:
            pred = model(images.to(device))  # 得到预测结果 [N,C]
            loss = loss_function(pred, labels.to(device))

        accu_loss += loss.detach()
        pred_classes = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    if cam:
        return accu_loss.item() / (step + 1), accu_loss_pos.item() / (step + 1), accu_loss_neg.item() / (
                    step + 1), accu_loss_sup.item() / (step + 1), \
               accu_num.item() / sample_num
    else:
        return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

# 获取训练的参数
def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

   #  print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

# import os
# import sys
# import json
# import pickle
# import random
# import math
#
# import torch
# from tqdm import tqdm
#
# import matplotlib.pyplot as plt
#
#
# def read_split_data(root: str, json_data):,
#     random.seed(2024)  # 保证随机结果可复现
#     assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
#
#     # 遍历文件夹，一个文件夹对应一个类别
#     survival_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))] # 获取每个文件夹的相对路径名字
#
#     # 排序，保证各平台顺序一致
#     survival_class.sort()
#
#     # 用一个字典生成类别名称和对应的索引
#     class_indices = dict() # {'xxx':0}
#     for k, v in enumerate(survival_class):
#         class_indices[v] = k
#
#     json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4) # 将python对象转为json对象
#     with open('class_indices.json', 'w') as json_file:
#         json_file.write(json_str) # 将json对象写入文件中
#
#     train_images_path = []  # 存储训练集的所有图片路径
#     train_images_label = []  # 存储训练集图片对应索引信息
#
#     val_images_path = []  # 存储验证集的所有图片路径
#     val_images_label = []  # 存储验证集图片对应索引信息
#
#     every_class_num = []  # 存储每个类别的样本总数
#
#     # 遍历每个文件夹下的文件
#     for cla in survival_class: # L M S
#         cla_path = os.path.join(root, cla) # 路径拼接 # ./data/PIC/L
#         cases_path = os.listdir(cla_path) # ['Brats18_CBICA_AAG_1', 'Brats18_CBICA_AAL_1', ...]
#         images = []
#
#         '''
#         更新读取数据：按照病例文件夹读，依次读取一个文件夹下的所有图像数据，以病例为单位：
#         '''
#         for case in cases_path:
#             flair_image = [os.path.join(cla_path, case, i) for i in os.listdir(os.path.join(cla_path, case)) if 'flair' in i]
#             # 得到五张flair切片数据  每个模态和对应切片的数据配对
#             for flair_path in flair_image:
#                 single_slice_path = [flair_path]
#                 single_slice_path.append(flair_path.replace('flair', 't1ce'))
#                 single_slice_path.append(flair_path.replace('flair', 't2'))
#                 single_slice_path.append(flair_path.replace('flair', 't1'))
#                 images.append(single_slice_path) # 以病例为单位 再以单张切片为单位 每个病例得到五份slice
#
#         # print(images)
#         # print(len(images))
#
#         # 获取该类别对应的索引
#         image_class = class_indices[cla] # 获取类别索引
#
#         # 记录该类别的样本数量
#         every_class_num.append(len(images))
#
#         for imgs_path in images:
#             for img_path in imgs_path:
#                 if img_path.split('\\')[-2] in json_data:  # 如果该路径在采样的验证集样本中则存入验证集
#                     val_images_path.append(imgs_path) # 直接放入病例下的所有文件列表
#                     val_images_label.append(image_class) # 图片文件和标签一一对应
#                     break
#                 else:  # 否则存入训练集
#                     train_images_path.append(imgs_path)
#                     train_images_label.append(image_class)
#                     break
#
#     # print(train_images_path)
#     # print(val_images_path)
#     # print(len(train_images_label)) # 2880 每个训练集图片的路径对应一个标签
#     # print(len(val_images_label)) # 680
#
#     print("{} images were found in the dataset.".format(sum(every_class_num)))
#     print("{} images for training.".format(len(train_images_path)))
#     print("{} images for validation.".format(len(val_images_path)))
#     assert len(train_images_path) > 0, "number of training images must greater than 0."
#     assert len(val_images_path) > 0, "number of validation images must greater than 0."
#
#     plot_image = False
#
#     if plot_image:
#         # 绘制每种类别个数柱状图
#         plt.bar(range(len(survival_class)), every_class_num, align='center')
#         # 将横坐标0,1,2,3,4替换为相应的类别名称
#         plt.xticks(range(len(survival_class)), survival_class)
#         # 在柱状图上添加数值标签
#         for i, v in enumerate(every_class_num):
#             plt.text(x=i, y=v + 5, s=str(v), ha='center')
#         # 设置x坐标
#         plt.xlabel('image class')
#         # 设置y坐标
#         plt.ylabel('number of images')
#         # 设置柱状图的标题
#         plt.title('survival class distribution')
#         plt.show()
#
#     return train_images_path, train_images_label, val_images_path, val_images_label
#
#
# def plot_data_loader_image(data_loader):
#     batch_size = data_loader.batch_size
#     plot_num = min(batch_size, 4)
#
#     json_path = './class_indices.json'
#     assert os.path.exists(json_path), json_path + " does not exist."
#     json_file = open(json_path, 'r')
#     class_indices = json.load(json_file)
#
#     for data in data_loader:
#         images, labels = data
#         for i in range(plot_num):
#             # [C, H, W] -> [H, W, C]
#             img = images[i].numpy().transpose(1, 2, 0)
#             # 反Normalize操作
#             img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
#             label = labels[i].item()
#             plt.subplot(1, plot_num, i+1)
#             plt.xlabel(class_indices[str(label)])
#             plt.xticks([])  # 去掉x轴的刻度
#             plt.yticks([])  # 去掉y轴的刻度
#             plt.imshow(img.astype('uint8'))
#         plt.show()
#
#
# def write_pickle(list_info: list, file_name: str):
#     with open(file_name, 'wb') as f:
#         pickle.dump(list_info, f)
#
#
# def read_pickle(file_name: str) -> list:
#     with open(file_name, 'rb') as f:
#         info_list = pickle.load(f)
#         return info_list
#
# def compute_loss(pred, labels):
#     loss_function = torch.nn.CrossEntropyLoss()  # 交叉熵损失
#     loss_flair = loss_function(pred[0], labels)
#     loss_t1ce = loss_function(pred[1], labels)
#     loss_t1 = loss_function(pred[2], labels)
#     loss_t2 = loss_function(pred[3], labels)
#     loss_fusion = loss_function(pred[4], labels)
#
#     return (loss_flair + loss_t1ce + loss_t1 + loss_t2) * 0.3 + loss_fusion * 0.7
#
# # 训练一个epoch的类
# def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
#     model.train() # 开启训练
#
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#     accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
#
#     optimizer.zero_grad() # 先是AdamW优化器梯度清零
#
#     sample_num = 0 # 总的训练样本数？
#
#     data_loader = tqdm(data_loader, file=sys.stdout)
#
#     # 每个batch的数据
#     for step, data in enumerate(data_loader):
#
#         images, labels, clinic = data
#
#         sample_num += images.shape[0]
#
#         pred = model(images.to(device), clinic.to(device)) # 得到预测结果 五个[N,C]
#
#         pred_classes = torch.max(pred[-1], dim=1)[1] # 取最后一个多模态特征的预测结果作为网络最终的预测结果 就是预测类别
#
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()
#
#         loss = compute_loss(pred, labels.to(device))
#
#         loss.backward()
#
#         accu_loss += loss.detach()
#
#         # 作用是用于设置 tqdm 进度条的描述信息
#         data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
#             epoch,
#             accu_loss.item() / (step + 1),
#             accu_num.item() / sample_num,
#             optimizer.param_groups[0]["lr"]
#         )
#
#         if not torch.isfinite(loss):
#             print('WARNING: non-finite loss, ending training ', loss)
#             sys.exit(1)
#
#         optimizer.step() # 优化器更新
#         optimizer.zero_grad()
#         # update lr
#         lr_scheduler.step() # 学习率更新
#
#     # 返回每个epoch训练完毕的loss和每个epoch内平均准确率
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num
#
# # 验证
# @torch.no_grad()
# def evaluate(model, data_loader, device, epoch):
#
#     model.eval() # 开始验证模式
#
#     accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#
#     sample_num = 0
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     for step, data in enumerate(data_loader):
#         images, labels, clinic = data
#         sample_num += images.shape[0]
#
#         pred = model(images.to(device), clinic.to(device))
#         pred_classes = torch.max(pred[-1], dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()
#
#         loss = compute_loss(pred, labels.to(device))
#         accu_loss += loss
#
#         data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
#             epoch,
#             accu_loss.item() / (step + 1),
#             accu_num.item() / sample_num
#         )
#
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num
#
#
# def create_lr_scheduler(optimizer,
#                         num_step: int,
#                         epochs: int,
#                         warmup=True,
#                         warmup_epochs=1,
#                         warmup_factor=1e-3,
#                         end_factor=1e-6):
#     assert num_step > 0 and epochs > 0
#     if warmup is False:
#         warmup_epochs = 0
#
#     def f(x):
#         """
#         根据step数返回一个学习率倍率因子，
#         注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
#         """
#         if warmup is True and x <= (warmup_epochs * num_step):
#             alpha = float(x) / (warmup_epochs * num_step)
#             # warmup过程中lr倍率因子从warmup_factor -> 1
#             return warmup_factor * (1 - alpha) + alpha
#         else:
#             current_step = (x - warmup_epochs * num_step)
#             cosine_steps = (epochs - warmup_epochs) * num_step
#             # warmup后lr倍率因子从1 -> end_factor
#             return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor
#
#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
#
# # 获取训练的参数
# def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
#     # 记录optimize要训练的权重参数
#     parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
#                             "no_decay": {"params": [], "weight_decay": 0.}}
#
#     # 记录对应的权重名称
#     parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
#                              "no_decay": {"params": [], "weight_decay": 0.}}
#
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue  # frozen weights
#
#         if len(param.shape) == 1 or name.endswith(".bias"):
#             group_name = "no_decay"
#         else:
#             group_name = "decay"
#
#         parameter_group_vars[group_name]["params"].append(param)
#         parameter_group_names[group_name]["params"].append(name)
#
#    #  print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
#     return list(parameter_group_vars.values())
