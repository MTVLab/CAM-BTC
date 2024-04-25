from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset): # 自定义数据集 继承自Dataset 实现三个方法
    '''自定义数据集'''

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path # 图像路径列表
        self.images_class = images_class # 图像标签列表
        self.transform = transform # transform措施

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))

        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        # 将标签变为独热编码
        # label = torch.nn.functional.one_hot(label, num_classes=5)
        # print(f'label:{label.shape}')
        path = self.images_path[item]
        # print(path)
        return img, label, path# 返回图像数据和标签

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels, path = tuple(zip(*batch))

        # print('batch:' , path)
        images = torch.stack(images, dim=0)
        path = list(path)
        labels = torch.as_tensor(labels)

        # labels = torch.nn.functional.one_hot(labels, num_classes=5)
        # print(labels)

        return images, labels,
