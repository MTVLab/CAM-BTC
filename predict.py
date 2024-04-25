from PIL import Image
import matplotlib.pyplot as plt

import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import convnext_tiny as create_model
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate
import numpy as np
import random
import json
import os
import sys
import json
import pickle
import random
import math
import numpy as np

import torch
from tqdm import tqdm
from compare.resnet_model import resnet18, resnet34
from compare.vgg_model import vgg
from compare.regnet_model import create_regnet
from compare.dense_model import densenet121
from compare.shufflle_model import shufflenet_v2_x0_5
from compare.effifientNet_model import efficientnet_b0
from compare.model_v2 import MobileNetV2


import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools


def setup_seed(seed): # 设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(model, data_loader, device, cam):

    model.eval() # 开始验证模式

    labels_total =[]
    pred_total = []
    pred_total_prob = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data

        if cam:
            pred, pred2, pred3 = model(images.to(device))  # 得到预测结果 [N,C]
        else:
            pred = model(images.to(device))  # 得到预测结果 [N,C]

        pred_classes = torch.max(pred, dim=1)[1]  # [1]是取最大值所在的索引 得到[B, index] [1]取的是索引表示每个样本的预测索引 即标签

        labels_total += labels[:].cpu()

        # 视为二分类任务 将预测标签和真实标签相同的

        pred_total += pred_classes[:].cpu()

        pred_total_prob += pred_classes[:].cpu()

    labels_total = [i.item() for i in labels_total]

    pred_total = [i.item() for i in pred_total]

    # pltMatrix(labels_total, pred_total)

    # with open('./figs/SLFICAM_label.txt', 'w') as f:
    #     for item in labels_total:
    #         f.write(str(item))  # 每个元素占一行，\n表示换行
    #         f.write("\n")
    #
    # with open('./figs/SLFICAM_pred.txt', 'w') as f:
    #     for item in pred_total:
    #         f.write(str(item))  # 每个元素占一行，\n表示换行
    #         f.write("\n")

    accuracy = accuracy_score(labels_total, pred_total)

    # print('直接计算精确率：', precision)
    precision = precision_score(labels_total, pred_total, average='macro')

    # 计算召回率
    recall = recall_score(labels_total, pred_total, average='macro')

    # 计算F1分数
    f1 = f1_score(labels_total, pred_total, average='macro')

    return accuracy, precision, recall, f1



def wtiteErrorCase(labels_total, pred_total, save_path, paths):
    for i in range(len(labels_total)):
        if labels_total[i] != pred_total[i]:
            temp = {}
            temp['true'] = labels_total[i]
            temp['pred'] = pred_total[i]
            temp['path'] = paths[i]
            save_path.append(temp)

    with open('./figs/error_path2.txt', 'w') as f:
        for item in save_path:
            f.write(f"true: {item['true']}, pred: {item['pred']}, path: {item['path']}")  # 每个元素占一行，\n表示换行
            f.write("\n")

    with open('./figs/pred.txt', 'w') as f:
        for item in pred_total:
            f.write(f"{item}\n")  # 每个元素占一行，\n表示换行

def pltBar():
    label = []
    ABN = []
    LFICAM = []
    SLFICAM = []
    ConvNext = []

    with open('./figs/LFICAM_label.txt', 'r') as f:
        for line in f:
            label.append(int(line.strip()))

    with open('./figs/SLFICAM_pred.txt', 'r') as f:
        for line in f:
            SLFICAM.append(int(line.strip()))

    with open('./figs/ABN_pred.txt', 'r') as f:
        for line in f:
            ABN.append(int(line.strip()))

    with open('./figs/LFICAM_pred.txt', 'r') as f:
        for line in f:
            LFICAM.append(int(line.strip()))

    with open('./figs/ConvNext_pred.txt', 'r') as f:
        for line in f:
            ConvNext.append(int(line.strip()))

    # 计算模型预测的正确和错误数量
    def calc_model_stats(true_labels, model_predictions):
        correct_counts = np.zeros(5, dtype=int)
        incorrect_counts = np.zeros(5, dtype=int)
        for true_label, pred_label in zip(true_labels, model_predictions):
            if true_label == pred_label:
                correct_counts[true_label] += 1
            else:
                incorrect_counts[true_label] += 1
        return correct_counts, incorrect_counts

    # 分别计算三个模型的统计数据

    model0 = calc_model_stats(label, ConvNext)
    model1 = calc_model_stats(label, ABN)
    model2 = calc_model_stats(label, LFICAM)
    model3 = calc_model_stats(label, SLFICAM)

    mapping = {0:'脑转移瘤', 1:'脑胶质瘤', 2:'脑膜瘤', 3:'正常图像', 4:'垂体瘤'}
    # 绘制每个类别对于所有模型的统计图
    def plot_stats_for_class(class_index):
        labels = ['ConvNext', 'ConvNext+ABN', 'ConvNext+LFICAM', 'Ours']
        correct_counts = [model0[0][class_index], model1[0][class_index], model2[0][class_index], model3[0][class_index]]
        incorrect_counts = [-model0[1][class_index], -model1[1][class_index], -model2[1][class_index], -model3[1][class_index]]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        fig, ax = plt.subplots()
        rects1 = ax.bar(x, correct_counts, width,  label='', color='green')
        rects2 = ax.bar(x, incorrect_counts, width, label='', color='red')

        ax.set_ylabel('个数', fontsize=14)

        ax.set_title(f'{mapping[class_index]}分类统计', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=14)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        plt.tight_layout()
        plt.savefig(f'./figs/{mapping[class_index]}分类统计.jpg', format='jpg', dpi=300)
        plt.show()

    # 绘制每个类别的统计图
    for class_index in range(5):
        plot_stats_for_class(class_index)

def pltMatrix(labels_total, pred_total):
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(labels_total, pred_total)

    # 类别名称
    class_names = ['BM', 'GBM', 'MNG', 'NORM', 'PIT']

    # 使用Seaborn绘制混淆矩阵的热力图
    plt.figure(figsize=(10, 8))  # 设置图像大小

    # 设置中文字体，这里使用系统中已有的SimHei字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 绘制混淆矩阵图像
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()

    # 设置坐标轴刻度和标签
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # 在每个单元格中添加数字
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()

    # 保存图像为高清JPG文件
    plt.savefig('./figs/Ours混淆矩阵.jpg', format='jpg', dpi=300)

    # 显示图像
    plt.show()

def matrix(model, data_loader, device, cam):
    model.eval()  # 开始验证模式

    labels_total = []
    pred_total = []
    paths = []
    save_path = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data

        if cam:
            pred, pred2, pred3 = model(images.to(device))  # 得到预测结果 [N,C]
        else:
            pred = model(images.to(device))  # 得到预测结果 [N,C]

        pred_classes = torch.max(pred, dim=1)[1]  # [1]是取最大值所在的索引 得到[B, index] [1]取的是索引表示每个样本的预测索引 即标签

        labels_total += labels[:].cpu()

        # 视为二分类任务 将预测标签和真实标签相同的

        pred_total += pred_classes[:].cpu()

        # paths += path[:]

    labels_total = [i.item() for i in labels_total]

    pred_total = [i.item() for i in pred_total]


    with open('./figs/SLFICAM_label.txt', 'w') as f:
        for item in labels_total:
            f.write(str(item))  # 每个元素占一行，\n表示换行
            f.write("\n")

    with open('./figs/SLFICAM_pred.txt', 'w') as f:
        for item in pred_total:
            f.write(str(item))  # 每个元素占一行，\n表示换行
            f.write("\n")


def main(args, valid_fold, log_path):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"using {device} device.")

    setup_seed(2024)

    with open(args.json_file, 'r') as json_file:
        json_data = json.load(json_file)  # 转为python对象

    json_data = json_data[valid_fold]

    # 得到所有的训练集图像，训练集标签，验证集图像，验证集标签
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path, json_data)

    img_size = 224 # 图像大小

    data_transform = {

        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),  # 随机中心裁剪
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化验证数据集 生成DataSet
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"]
                            )

    # batch-size
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn
                                             )

    model = create_model(num_classes=args.num_classes).to(device)

    # model = densenet121(num_classes=args.num_classes).cuda()
    # model = resnet18(num_classes=args.num_classes, cam=args.cam).cuda() #
    # model = vgg(num_classes=args.num_classes).cuda() #
    # model = shufflenet_v2_x0_5(num_classes=args.num_classes).cuda() #
    # model = create_regnet(num_classes=args.num_classes).cuda() #
    # model = efficientnet_b0(num_classes=args.num_classes).cuda()
    # model = MobileNetV2(num_classes=args.num_classes).cuda()

    weights_dict = torch.load(args.weights, map_location=device)

    model.load_state_dict(weights_dict, strict=False) # 加载预训练权重

    model.eval()

    accuracy, precision, recall, f1score = evaluate(model=model,
                                                    data_loader=val_loader,
                                                    device=device,
                                                    cam=args.cam
                                                    )

    with open(os.path.join(log_path, "pred_log.txt"), "a+") as f:
        f.write(f"acc:{accuracy}, precision:{precision}, recall:{recall}, f1:{f1score}")
        f.write("\n")

    # matrix(model, val_loader, device, args.cam)

if __name__ == '__main__':

    parser = argparse.ArgumentParser() # 获取参数配置器
    parser.add_argument('--num_classes', type=int, default=5) # 类别
    parser.add_argument('--batch-size', type=int, default=32) # batch_size大小
    parser.add_argument('--cam', default=True)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="./data/PIC") # 数据集目录
    parser.add_argument('--json_file', type=str, default="./data/Fold10.json")  # 数据集目录

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default=None, help='initial weights path')

    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args() # 解析命令行参数并配置到命名空间

    # pltBar()

    for i in ['8']:
        log_path = f'runs/ConvNext_SLFICAM/Mar13_10-49-18_Stage4_8/fold_{i}'
        opt.weights = log_path + '/best_model.pth'
        main(opt, i, log_path)

