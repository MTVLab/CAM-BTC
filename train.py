import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision.models import squeezenet

from my_dataset import MyDataSet
from model import convnext_tiny as create_model
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate
import numpy as np
from datetime import datetime
import random
import json
from compare.resnet_model import resnet18, resnet34
from compare.vgg_model import vgg
from compare.regnet_model import create_regnet
from compare.dense_model import densenet121
from compare.shufflle_model import shufflenet_v2_x0_5
from compare.effifientNet_model import efficientnet_b0
from compare.model_v2 import MobileNetV2

def setup_seed(seed): # 设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args, valid_fold, log_path):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    setup_seed(2024)

    with open(args.json_file, 'r') as json_file:
        json_data = json.load(json_file)  # 转为python对象

    json_data = json_data[valid_fold]

    fold_log_path = log_path + f'/fold_{valid_fold}'
    if os.path.exists(fold_log_path) is False:
        os.makedirs(fold_log_path)

    tb_writer = SummaryWriter(fold_log_path) # 记录信息的对象 方便使用Tensorboard进行可视化

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 得到所有的训练集图像，训练集标签，验证集图像，验证集标签
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path, json_data)

    img_size = 224 # 图像大小

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),# 随机缩放裁剪
                                     # transforms.RandomHorizontalFlip(),# 随机水平反转
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),# 随机中心裁剪
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集 传入数据路径和标签列表，还有data.transformer
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"]
                              )

    # 实例化验证数据集 生成DataSet
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"]
                            )

    # batch-size
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    print('Using {} dataloader workers every process'.format(nw))

    # 生成DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn
                                             )

    # 传入num_classes 创建模型
    model = create_model(num_classes=args.num_classes).to(device)

    # model = densenet121(num_classes=args.num_classes).cuda()
    # model = resnet18(num_classes=args.num_classes, cam=args.cam).cuda() #
    # model = vgg(num_classes=args.num_classes).cuda() #
    # model = create_regnet(num_classes=args.num_classes).cuda() #

    # model = shufflenet_v2_x0_5(num_classes=args.num_classes).cuda() #
    # model = efficientnet_b0(num_classes=args.num_classes).cuda()
    # model = MobileNetV2(num_classes=args.num_classes).cuda()

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)

    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd) # AdamW优化器

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1) # 学习率调度器

    best_acc = 0.
    # minimum_val_loss = np.inf
    max_val_acc = -np.inf

    epochs_without_improvement = 0
    a = torch.nn.Parameter(torch.tensor(1.0))
    b = torch.nn.Parameter(torch.tensor(1.0))

    for epoch in range(args.epochs):
        # train 返回每个epoch训练完的loss和准确率
        tags = ["train_loss", "train_loss_cl", "train_loss_am", "train_loss_ax", "train_acc",
                "val_loss", "val_loss_cl", "val_loss_am", "val_loss_ax", "val_acc", "learning_rate"]

        if args.cam:
            train_loss, train_loss_cl, train_loss_am, train_loss_ax, train_acc = train_one_epoch(model=model,
                                                                                                 optimizer=optimizer,
                                                                                                 data_loader=train_loader,
                                                                                                 device=device,
                                                                                                 epoch=epoch,
                                                                                                 lr_scheduler=lr_scheduler,
                                                                                                 cam=args.cam,
                                                                                                 )

            val_loss, val_loss_cl, val_loss_am, val_loss_ax, val_acc = evaluate(model=model,
                                                                                data_loader=val_loader,
                                                                                device=device,
                                                                                epoch=epoch,
                                                                                cam=args.cam,
                                                                                )
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_loss_cl, epoch)
            tb_writer.add_scalar(tags[2], train_loss_am, epoch)
            tb_writer.add_scalar(tags[3], train_loss_ax, epoch)
            tb_writer.add_scalar(tags[4], train_acc, epoch)

            tb_writer.add_scalar(tags[5], val_loss, epoch)
            tb_writer.add_scalar(tags[6], val_loss_cl, epoch)
            tb_writer.add_scalar(tags[7], val_loss_am, epoch)
            tb_writer.add_scalar(tags[8], val_loss_ax, epoch)
            tb_writer.add_scalar(tags[9], val_acc, epoch)

        else:
            train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    lr_scheduler=lr_scheduler,
                                                    cam=args.cam
                                                    )

            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch,
                                         cam=args.cam
                                         )

            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[4], train_acc, epoch)

            tb_writer.add_scalar(tags[5], val_loss, epoch)
            tb_writer.add_scalar(tags[9], val_acc, epoch)

        tb_writer.add_scalar(tags[9], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            torch.save(model.state_dict(), fold_log_path + f"/best_model.pth") # 保存模型权重
            best_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # 早停策略：当验证集上的损失连续x个epoch都没有降低，则停止训练
        # if minimum_val_loss > val_loss:
        #     minimum_val_loss = val_loss
        #     epochs_without_improvement = 0
        # else:
        #     epochs_without_improvement += 1

        if epochs_without_improvement == args.early_stop_epochs:
            print(f'early_stop_epoch:{epoch}, best_val_acc:{best_acc}')
            break

        with open(os.path.join(fold_log_path, "log.txt"), "a+") as f:
            f.write(f"epoch:{epoch + 1}, train_acc:{train_acc}, val_acc:{val_acc}")
            f.write("\n")
            f.write(f"epoch:{epoch + 1}, best_val_acc:{best_acc}")
            f.write("\n")
            f.write("*" * 20)
            f.write("\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser() # 获取参数配置器
    parser.add_argument('--num_classes', type=int, default=5) # 类别
    parser.add_argument('--epochs', type=int, default=200) # 训练轮次
    parser.add_argument('--batch-size', type=int, default=64) #
    parser.add_argument('--lr', type=float, default=5e-4) # 学习率
    parser.add_argument('--wd', type=float, default=5e-2) # 权重衰退
    parser.add_argument('--early_stop_epochs', type=int, default=200) # 早停策略
    parser.add_argument('--cam', default=True)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="./data/PIC") # 数据集目录
    parser.add_argument('--json_file', type=str, default="./data/Fold10.json")  # 数据集目录

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='', help='initial weights path')

    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args() # 解析命令行参数并配置到命名空间

    log_path = './runs/ConvNext_SLFICAM/' + datetime.now().strftime("%b%d_%H-%M-%S")

    for i in ['8']:
        main(opt, i, log_path)
